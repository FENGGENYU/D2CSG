import torch.nn as nn
import torch
import torch.nn.functional as F

FLOAT_EPS = torch.finfo(torch.float32).eps

class Myminclamp(torch.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx, input):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		ctx.save_for_backward(input)
		return input.clamp(min=0)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		#grad_input[input < 0] = 0.001*grad_input[input < 0].clamp(max=0)
		grad_input = grad_input*(input >= 0).float() + 0.001* (grad_input * (input < 0).float()).clamp(max=0)
		return grad_input

class Mymaxclamp(torch.autograd.Function):
	"""
	We can implement our own custom autograd Functions by subclassing
	torch.autograd.Function and implementing the forward and backward passes
	which operate on Tensors.
	"""

	@staticmethod
	def forward(ctx, input):
		"""
		In the forward pass we receive a Tensor containing the input and return
		a Tensor containing the output. ctx is a context object that can be used
		to stash information for backward computation. You can cache arbitrary
		objects for use in the backward pass using the ctx.save_for_backward method.
		"""
		ctx.save_for_backward(input)
		return input.clamp(max=1)#torch.clamp(x, max=1)

	@staticmethod
	def backward(ctx, grad_output):
		"""
		In the backward pass we receive a Tensor containing the gradient of the loss
		with respect to the output, and we need to compute the gradient of the loss
		with respect to the input.
		"""
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		#grad_input[input > 1] = 0.001*grad_input[input > 1].clamp(min=0)
		grad_input = grad_input *(input <= 1).float() + 0.001*(grad_input * (input > 1).float()).clamp(min=0)
		return grad_input

#produce convex
class Generator(nn.Module):
	def __init__(self, p_dim=32, c_dim=8):
		super(Generator, self).__init__()
		self.p_dim = p_dim
		self.c_dim = c_dim
		self.half_p_dim = int(p_dim/2)
		self.half_c_dim = int(c_dim/2)
		convex_layer_weights = torch.zeros((self.half_p_dim, self.half_c_dim))
		concave_layer_weights = torch.zeros((self.half_p_dim, self.half_c_dim))
		union_layer_weights = torch.zeros((self.c_dim, 1))
		self.unused_convex = torch.zeros(self.c_dim).cuda() #use 0, unsed 1

		self.convex_layer_weights = nn.Parameter(convex_layer_weights)
		self.concave_layer_weights = nn.Parameter(concave_layer_weights)
		self.union_layer_weights = nn.Parameter(union_layer_weights)
		#self.unused_convex = nn.Parameter(unused_convex)

		nn.init.normal_(self.convex_layer_weights, mean=0.0, std=0.02)
		nn.init.normal_(self.concave_layer_weights, mean=0.0, std=0.02)
		nn.init.normal_(self.union_layer_weights, mean=1e-5, std=0.02)

	def forward(self, surface_points_normals, near_surface_points, primitives, phase, is_leaky=False):
		min_clamp = Myminclamp.apply
		max_clamp = Mymaxclamp.apply
		#matrix preparation
		N_ones = torch.ones(near_surface_points.size(0), near_surface_points.size(1), 1).cuda()
		N_pointsx = torch.cat([near_surface_points**2, near_surface_points, N_ones], 2)
		assign_matrix_N_P = None
		if phase==0: #S+
			#level 1 D=relu(xP)
			#points b*x*4
			#planes b*4*P
			#D b*x*P, inside 0, outside 1
			h1 = torch.matmul(N_pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)
			#level 2 C=1-DT
			#C b*x*C, inside 1, outside 0
			h2_convex = torch.matmul(h1[:, :, :self.half_p_dim], self.convex_layer_weights)
			h2_concave = torch.matmul(h1[:, :, self.half_p_dim:], self.concave_layer_weights)
			if is_leaky:
				h2_convex = max_clamp(min_clamp(1 - h2_convex))
				h2_concave = max_clamp(min_clamp(1 - h2_concave))
			else:
				h2_convex = torch.clamp(1-h2_convex, min=0, max=1)
				h2_concave = torch.clamp(1-h2_concave, min=0, max=1)
			#level 3 S+=CW
			#S b*x*1

			h3_1 = torch.matmul(h2_convex, self.union_layer_weights[:self.half_c_dim, :])
			h3_0 = torch.matmul(h2_concave, self.union_layer_weights[self.half_c_dim:, :])
			#h3_1 inside 1 outside 0
			#1 - h3_0 inside 0 outside 1
			h3 = torch.min(h3_1, (1-h3_0))
			h2 = torch.cat([h2_convex, h2_concave], dim=2)
			return assign_matrix_N_P, h3_1.squeeze(-1), h3_0.squeeze(-1), h3.squeeze(-1), h2
		elif phase==1:
			h1 = torch.matmul(N_pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)
			#level 2 C=1-DT
			#C b*x*C, inside 1, outside 0
			h2_convex = torch.matmul(h1[:, :, :self.half_p_dim], self.convex_layer_weights)
			h2_concave = torch.matmul(h1[:, :, self.half_p_dim:], self.concave_layer_weights)

			h3_1 = torch.min(h2_convex, dim=2)[0]
			h3_0 = torch.min(h2_concave, dim=2)[0]
			h3 = 1 - torch.max(h3_1, 0.2-h3_0)
			#h3 = torch.min((h3_1 < 0.1).float(), 1-(h3_0 < 0.1).float())
			#h3 = 1- h3
			h2 = torch.cat([h2_convex, h2_concave], dim=2)
			h2 = (1-h2)
			return assign_matrix_N_P, h3_1, h3_0, h3, h2
		elif phase==2:
			h1 = torch.matmul(N_pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)
			#level 2 C=1-DT
			#C b*x*C, inside 1, outside 0
			h2_convex = torch.matmul(h1[:, :, :self.half_p_dim],  (self.convex_layer_weights>0.01).float())
			h2_concave = torch.matmul(h1[:, :, self.half_p_dim:], (self.concave_layer_weights>0.01).float())
			#print((self.concave_layer_weights>0.01).float())
			#level 2 C=1-DT
			#C b*x*C, inside 0, outside 0-1
			h3_1 = torch.min(h2_convex, dim=2)[0]
			h3_0 = torch.min(h2_concave, dim=2)[0]

			#h3_1 inside 0, outside 0+
			#h3_0 inside 0, outside 0+
			#0.2 - h3_0 inside 0.2 outside 0.2-
			if torch.sum((h3_0 > 0.01).float()) > 0:
				#h3 = torch.min((h3_1 < 0.01).float(), 1-(h3_0 < 0.01).float())
				h3 = 1 - torch.max(h3_1, 0.2-h3_0)
				h2 = torch.cat([h2_convex, h2_concave], dim=2)
			else:
				h3 = (h3_1 < 0.01).float()
				h2 = torch.cat([h2_convex, torch.ones(h2_concave.size()).cuda()], dim=2)

			#h3 = 1 - h3
			h2 = (1-h2)
			return assign_matrix_N_P, h3_1, h3_0, h3, h2
		elif phase==3:
			h1 = torch.matmul(N_pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)
			#level 2 C=1-DT
			#C b*x*C, inside 1, outside 0
			#assert convex_mask is not None

			convex_mask = self.unused_convex.reshape(1,1,-1)
			h2_convex = torch.matmul(h1[:, :, :self.half_p_dim],  (self.convex_layer_weights>0.01).float())
			h2_concave = torch.matmul(h1[:, :, self.half_p_dim:], (self.concave_layer_weights>0.01).float())
			h2_convex = h2_convex + (convex_mask[:, :, :self.half_c_dim])*100
			h2_concave = h2_concave + (convex_mask[:, :, self.half_c_dim:])*100
			#level 2 C=1-DT
			#C b*x*C, inside 0, outside 0-1
			h3_1 = torch.min(h2_convex, dim=2)[0]
			h3_0 = torch.min(h2_concave, dim=2)[0]

			#h3_1 inside 0, outside 0+
			#h3_0 inside 0, outside 0+
			#0.2 - h3_0 inside 0.2 outside 0.2-
			#torch.max(h3_1, 0.2-h3_0) in 0, outsie 0+
			if torch.sum((h3_0 > 0.01).float()) > 0:
				#h3 = torch.max(h3_1, 0.2-h3_0)
				#h3 = torch.min((h3_1 < 0.01).float(), (h3_0 > 0.01).float())
				#h3 = (h3_1 < 0.01).float()
				h3 = 1 - torch.max(h3_1, 0.2-h3_0)
				h2 = torch.cat([h2_convex, h2_concave], dim=2)
			else:
				h3 = (h3_1 < 0.01).float()
				h2 = torch.cat([h2_convex, torch.ones(h2_concave.size()).cuda()], dim=2)
			#assert torch.max(h3) < 2
			#h3 = 1 - h3
			h2 = (1-h2)
			return assign_matrix_N_P, h3_1, h3_0, h3, h2
		elif phase==4:
			h1 = torch.matmul(N_pointsx, primitives)
			if is_leaky:
				h1 = min_clamp(h1)
			else:
				h1 = torch.clamp(h1, min=0)
			#level 2 C=1-DT
			#C b*x*C, inside 0, outside 0-1
			# selected_1 = [41]
			# selected_2 = [201-128, 233-128]
			# mask1 = torch.zeros((self.half_p_dim, self.half_c_dim))
			# mask1[:, selected_1] = 1
			# #P, C
			# mask1 = (self.convex_layer_weights>0.01).float() * mask1.cuda()

			# mask2 = torch.zeros((self.half_p_dim, self.half_c_dim))
			# mask2[:, selected_2] = 1
			# mask2 = (self.concave_layer_weights>0.01).float() * mask2.cuda()
			# #print(torch.sum(mask2))
			# valid_primitives_1 = (torch.sum(mask1, dim=1) > 0).float().reshape(1, -1)
			# valid_primitives_2 = (torch.sum(mask2, dim=1) > 0).float().reshape(1, -1)
			# valid_primitives = torch.cat((valid_primitives_1, valid_primitives_2), dim=1)

			# h2_convex = torch.matmul(h1_convex, mask1)
			# h2_concave = torch.matmul(h1_concave, mask2)

			# h3_1 = torch.min(h2_convex[:, :, selected_1], dim=2)[0]
			# h3_0 = torch.min(h2_concave[:, :, selected_2], dim=2)[0]

			h2_convex = torch.matmul(h1[:, :, :self.half_p_dim],  (self.convex_layer_weights>0.01).float())
			h2_concave = torch.matmul(h1[:, :, self.half_p_dim:], (self.concave_layer_weights>0.01).float())
			#level 2 C=1-DT
			#C b*x*C, inside 0, outside 0-1
			h3_1 = torch.min(h2_convex, dim=2)[0]
			h3_0 = torch.min(h2_concave, dim=2)[0]

			#h3_1 inside 0, outside 0+
			#h3_0 inside 0, outside 0+
			#0.2 - h3_0 inside 0.2 outside 0.2-
			h3 = torch.max(h3_1, 0.2-h3_0)
			h3 = 1 - h3
			h2 = torch.cat([h2_convex, h2_concave], dim=2)
			h2 = (1-h2)
			return assign_matrix_N_P, h3_1, h3_0, h3, h2
		else:
			print("Congrats you got an error!")
			print("generator.phase should be in [0,1,2,3], got", phase)
			exit(0)

class Encoder(nn.Module):
	def __init__(self, ef_dim=32):
		super(Encoder, self).__init__()
		self.ef_dim = ef_dim
		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=True)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=True)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=True)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*8, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.constant_(self.conv_1.bias,0)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.constant_(self.conv_2.bias,0)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.constant_(self.conv_3.bias,0)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.constant_(self.conv_4.bias,0)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)


	def forward(self, inputs, is_training=False):
		d_1 = self.conv_1(inputs)
		d_1 = F.leaky_relu(d_1, negative_slope=0.01, inplace=True)

		d_2 = self.conv_2(d_1)
		d_2 = F.leaky_relu(d_2, negative_slope=0.01, inplace=True)
		
		d_3 = self.conv_3(d_2)
		d_3 = F.leaky_relu(d_3, negative_slope=0.01, inplace=True)
		#print('d_3', d_3.shape)
		d_4 = self.conv_4(d_3)#32 256 
		#print('d_4', d_4.shape)
		d_4 = F.leaky_relu(d_4, negative_slope=0.01, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = d_5.view(-1, self.ef_dim*8)
		d_5 = F.leaky_relu(d_5, negative_slope=0.01, inplace=True)

		return d_5

class Decoder(nn.Module):
	def __init__(self, ef_dim=32, p_dim=32):
		super(Decoder, self).__init__()
		self.ef_dim = ef_dim
		self.p_dim = p_dim
		self.linear_1 = nn.Linear(self.ef_dim*8, self.ef_dim*16, bias=True)
		self.linear_2 = nn.Linear(self.ef_dim*16, self.p_dim*7, bias=True)

		nn.init.xavier_uniform_(self.linear_1.weight)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.xavier_uniform_(self.linear_2.weight)
		nn.init.constant_(self.linear_2.bias,0)

	def forward(self, inputs, pretrain=False):
		l1 = self.linear_1(inputs)
		l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)

		l2 = self.linear_2(l1)
		l2 = l2.view(-1, 7, int(self.p_dim))
		#first half constraints
		l3 = torch.cat([torch.abs(l2[:, :3, :int(self.p_dim/4)]), l2[:, 3:, :int(self.p_dim/4)]], 1)
		l4 = torch.cat([-1*torch.abs(l2[:, :3, int(self.p_dim/4):int(self.p_dim/2)]), l2[:, 3:, int(self.p_dim/4):int(self.p_dim/2)]], 1)
		#second half no constraints
		#l4 = l2[:, :, int(self.p_dim/2):]
		l5 = torch.cat([torch.abs(l2[:, :3, int(self.p_dim/2):int(self.p_dim*3/4)]), l2[:, 3:, int(self.p_dim/2):int(self.p_dim*3/4)]], 1)
		l6 = torch.cat([-1*torch.abs(l2[:, :3, int(self.p_dim*3/4):]), l2[:, 3:, int(self.p_dim*3/4):]], 1)
		l7 = torch.cat([l3, l4, l5, l6], dim=2)
		return l7
