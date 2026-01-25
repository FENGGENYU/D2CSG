import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_composite_loss_compact(plane_convex, shapenet_flag, phase, point_convex, \
	preds, G, G_a, trues, shape_layer, convex_layer=None, concave_layer=None, \
	no_weight=False, primitives=None, sym_planes = None):
	#total_loss = 0
	if no_weight:
		a = 1
		b = 1
		c = 1
	else:
		a = 10
		b = 2.5
		c = 1

	l1_loss = torch.Tensor([0])
	norm_05 = torch.Tensor([0])
	if phase==0:
		#phase 0 continuous for better convergence
		#L_recon + L_W + L_T
		#G2 - network output (convex layer), the last dim is the number of convexes
		#G - network output (final output)
		#point_value - ground truth inside-outside value for each point
		
		inversed = 1-trues
		mask_1 = torch.max(trues, (G_a<0.5).float())
		mask_2 = torch.max(trues, (G>0.5).float())
		loss_sp = torch.mean(((trues*(1-G.clamp(max=1)) + inversed*(G.clamp(min=0)))**2) *mask_1)
		loss_sp_a = torch.mean(((inversed*(1-G_a.clamp(max=1)) + trues*(G_a.clamp(min=0)))**2) *mask_2)

		total_loss = loss_sp + loss_sp_a + torch.sum(torch.abs(shape_layer-1))
		if convex_layer is not None:
			total_loss +=  (torch.sum(torch.clamp(convex_layer-1, min=0) - torch.clamp(convex_layer, max=0)))
			total_loss +=  (torch.sum(torch.clamp(concave_layer-1, min=0) - torch.clamp(concave_layer, max=0)))
			#total_loss +=  0.001*(torch.sum((convex_layer)**2))
		if plane_convex is not None:
			#norm_05 =  0.01*(torch.sum(torch.exp(-1000 * torch.clamp(plane_convex, max=0.01)))/(torch.sum((plane_convex < 0.01).float()) + 1))
			norm_05 =  0.01*(torch.sum(torch.exp(-10 * plane_convex.clamp(max=1) ))/(torch.sum((plane_convex < 0.01).float()) + 1))
			
			total_loss += norm_05
		if sym_planes is not None:
			a=sym_planes[:, 0]
			b=sym_planes[:, 1]
			c=sym_planes[:, 2]
			d=sym_planes[:, 3]
			distance = torch.abs(d)/(a**2+b**2+c**2)**0.5
			l1_loss = torch.sum(torch.clamp(distance-0.2, min=0))
			total_loss += l1_loss
	elif phase==1:
		#phase 1 hard discrete for bsp
		#L_recon
		inversed = 1-trues
		# mask_1: 1 must be 1, 0 can be masked wrt self.G_a
		mask_1 = torch.max(trues, (G_a>0.01).float())
		# mask_2: (inversed) 0 must be 0, 1 can be masked wrt self.G
		mask_2 = torch.max(trues, (G<0.01).float())
		
		loss_sp = torch.mean((inversed*(1-G.clamp(max=1)) + a*trues*(G.clamp(min=0)))*mask_1)
		loss_sp_a = torch.mean((trues*(1-G_a.clamp(max=1))*c + b*inversed*(G_a.clamp(min=0)))*mask_2)
		total_loss = loss_sp + loss_sp_a
		if convex_layer is not None:
			total_loss +=  (torch.sum(torch.clamp(convex_layer-1, min=0) - torch.clamp(convex_layer, max=0)))
			total_loss +=  (torch.sum(torch.clamp(concave_layer-1, min=0) - torch.clamp(concave_layer, max=0)))
			#total_loss +=  0.001*(torch.sum((convex_layer)**2))
		# l1_loss =  0.1*(torch.mean((planes[:, :3, :])**2) + torch.mean(torch.exp(-1e2 * torch.abs(planes[:, 3:, :]))))
		# #print('l1 loss, ', l1_loss)
		# total_loss += l1_loss
		if plane_convex is not None:
			#norm_05 =  0.01*(torch.sum(torch.exp(-1000 * torch.clamp(plane_convex, max=0.01)))/(torch.sum((plane_convex < 0.01).float()) + 1))
			norm_05 =  0.01*(torch.sum(torch.exp(-10 * plane_convex.clamp(max=1) ))/(torch.sum((plane_convex < 0.01).float()) + 1))
			
			total_loss += norm_05
		if sym_planes is not None:
			a=sym_planes[:, 0]
			b=sym_planes[:, 1]
			c=sym_planes[:, 2]
			d=sym_planes[:, 3]
			distance = torch.abs(d)/(a**2+b**2+c**2)**0.5
			l1_loss = torch.sum(torch.clamp(distance-0.2, min=0))
			total_loss += l1_loss
	elif phase==2:
		#phase 1 hard discrete for bsp
		#L_recon
		inversed = 1-trues
		# mask_1: 1 must be 1, 0 can be masked wrt self.G_a
		mask_1 = torch.max(trues, (G_a>0.01).float())
		# mask_2: (inversed) 0 must be 0, 1 can be masked wrt self.G
		mask_2 = torch.max(trues, (G<0.01).float())
		
		loss_sp = torch.mean((inversed*(1-G.clamp(max=1)) + a*trues*(G.clamp(min=0)))*mask_1)
		loss_sp_a = torch.mean((trues*(1-G_a.clamp(max=1))*c + b*inversed*(G_a.clamp(min=0)))*mask_2)
		
		total_loss = loss_sp + loss_sp_a
		if plane_convex is not None:
			#norm_05 =  0.01*(torch.sum(torch.exp(-1000 * torch.clamp(plane_convex, max=0.01)))/(torch.sum((plane_convex < 0.01).float()) + 1))
			norm_05 =  0.01*(torch.sum(torch.exp(-10 * plane_convex.clamp(max=1) ))/(torch.sum((plane_convex < 0.01).float()) + 1))
			
			total_loss += norm_05
			
		if sym_planes is not None:
			a=sym_planes[:, 0]
			b=sym_planes[:, 1]
			c=sym_planes[:, 2]
			d=sym_planes[:, 3]
			distance = torch.abs(d)/(a**2+b**2+c**2)**0.5
			l1_loss = torch.sum(torch.clamp(distance-0.2, min=0))
			total_loss += l1_loss
		# #print('planes, ', planes)
		# l1_loss =  0.1*(torch.mean((planes[:, 3:, :])**2) + torch.mean(torch.exp(-1e2 * torch.abs(planes[:, :3, :]))))
		# #print('l1 loss, ', l1_loss)
		# total_loss += l1_loss
	elif phase==3:
		#phase 1 hard discrete for bsp
		#L_recon
		inversed = 1-trues
		# mask_1: 1 must be 1, 0 can be masked wrt self.G_a
		mask_1 = torch.max(trues, (G_a>0.01).float())
		# mask_2: (inversed) 0 must be 0, 1 can be masked wrt self.G
		mask_2 = torch.max(trues, (G<0.01).float())
		
		loss_sp = torch.mean((inversed*(1-G.clamp(max=1)) + a*trues*(G.clamp(min=0)))*mask_1)
		loss_sp_a = torch.mean((trues*(1-G_a.clamp(max=1))*c + b*inversed*(G_a.clamp(min=0)))*mask_2)
		
		total_loss = loss_sp + loss_sp_a
		# #print('planes, ', planes)
		#primi_mask = primi_mask.reshape(1, 1, -1)
		#primitives = torch.cat((planes, quads), 2)
		#l1_loss =  (torch.mean(torch.exp(-0.01 * torch.abs(planes*primi_mask[:512].reshape(1, -1, 1))))) + (torch.mean(torch.exp(-0.01 * torch.abs(quads))))
		#l1_loss =  (torch.mean(torch.exp(-0.01 * torch.abs(primitives))))
		
		# #print('l1 loss, ', l1_loss)
		#total_loss += l1_loss
		#total_loss += norm_05
	else:
		print('error....................')

	return norm_05, l1_loss, loss_sp, loss_sp_a, total_loss

