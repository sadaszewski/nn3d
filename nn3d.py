#
# Discrete 3D Natural Neighbor Approximation for OpenCL
# 
# Copyright (C) Stanislaw Adaszewski, 2016
# http://algoholic.eu
#


import numpy as np
import pyopencl as cl


class KDTreeNode:
	def __init__(self):
		self.ax = self.split = self.L = self.R = None
		self.n = None
		self.childCnt = 0
		
	def __eq__(self, other):
		return (self.ax == other.ax and \
			self.split == other.split and \
			np.all(self.L == other.L) and \
			np.all(self.R == other.R) and \
			self.n == other.n and \
			self.childCnt == other.childCnt)


def buildKDTree(pts, ax=0, maxPts=32):
	n = len(pts)
	ii = np.argsort(pts[:, 0])
	split = pts[ii[n/2-1], ax]
	L = pts[pts[:, ax] <= split, :]
	R = pts[pts[:, ax] > split, :]
	
	node = KDTreeNode()
	node.ax = ax
	node.split = split
	node.n = n
	
	if len(L) > maxPts:
		node.L = buildKDTree(L, (ax + 1) % 3, maxPts)
		node.childCnt += node.L.childCnt + 1
	else:
		node.L = L
		
	if len(R) > maxPts:
		node.R = buildKDTree(R, (ax + 1) % 3, maxPts)
		node.childCnt += node.R.childCnt + 1
	else:
		node.R = R
		
	return node
		

def flattenKDTree(node, out=None, ofs=None):
	# print 'ofs:', ofs
	if out is None:
		out = np.zeros(((node.childCnt + 1) * 6 + node.n * 4,), dtype=np.float32)
		return flattenKDTree(node, out, [0])
	else:
		if isinstance(node, KDTreeNode):
			# print 'node'
			out[ofs[0]] = node.ax
			out[ofs[0] + 1] = node.split
			out[ofs[0] + 2] = node.n
			out[ofs[0] + 3] = node.childCnt
			
			if isinstance(node.L, KDTreeNode):
				out[ofs[0] + 4] = -1
			else:
				out[ofs[0] + 4] = len(node.L)
				
			if isinstance(node.R, KDTreeNode):
				out[ofs[0] + 5] = -1
			else:
				out[ofs[0] + 5] = len(node.R)
			
			ofs[0] += 6
			flattenKDTree(node.L, out, ofs)
			flattenKDTree(node.R, out, ofs)
		else:
			# print 'adding points at', ofs[0], '(out of', len(out),') #points:', len(node)
			out[ofs[0]:ofs[0]+len(node)*4] = np.ravel(node)
			ofs[0] += len(node) * 4
		
		return out


def unflattenKDTree(flat, ofs=None):
	if ofs is None:
		ofs = [0]

	node = KDTreeNode()
	node.ax = flat[ofs[0]]
	node.split = flat[ofs[0] + 1]
	node.n = flat[ofs[0] + 2]
	node.childCnt = flat[ofs[0] + 3]
	
	nL = flat[ofs[0] + 4]
	nR = flat[ofs[0] + 5]
	
	ofs[0] += 6
	
	if nL >= 0: # left child is points
		node.L = np.reshape(flat[ofs[0]:ofs[0] + nL * 4], (nL, 4))
		ofs[0] += nL * 4
	else: # left child is node
		node.L = unflattenKDTree(flat, ofs)
	
	if nR >= 0:
		node.R = np.reshape(flat[ofs[0]:ofs[0] + nR * 4], (nR, 4))
		ofs[0] += nR * 4
	else:
		node.R = unflattenKDTree(flat, ofs)
		
	return node


def nn3d(pts, values,
	minima=np.zeros((3,), dtype=np.float32),
	maxima=np.ones((3,), dtype=np.float32),
	res=np.ones((3,), dtype=np.int32) * 32, ):
	
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	mf = cl.mem_flags

	pts2 = np.hstack((pts, np.reshape(values, (len(values),1))))
	print 'pts2:', pts2
	node = buildKDTree(pts2)
	flat = flattenKDTree(node)
	flat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = flat)
	
	minima_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = minima)
	maxima_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = maxima)
	res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = res)
	
	out = np.zeros(res, dtype=np.float32)
	out_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = out)
	
	# values_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = values.astype(np.float32))
	
	accum = np.zeros(res, dtype=np.float32)
	cnt = np.zeros(res, dtype=np.float32)
	
	accum_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=accum)
	cnt_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=cnt)
	
	prg = cl.Program(ctx, """
		void atomic_add_global(volatile __global float *source, const float operand) {
		
			// *source += operand;
			union {
				unsigned int intVal;
				float floatVal;
			} newVal;
			union {
				unsigned int intVal;
				float floatVal;
			} prevVal;

			do {
				prevVal.floatVal = *source;
				newVal.floatVal = prevVal.floatVal + operand;
			} while (atomic_cmpxchg((volatile global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
		}
		
		__global const float* closestPoint2(__global const float *flat_g, int cnt, float *pt) {
			float minDistSq = 1.0 / 0.0;
			int minIdx = -1;
			__global const float *p = flat_g;
		
			for (int i = 0; i < cnt; i++) {
				float x = *p++;
				float y = *p++;
				float z = *p++;
				float val = *p++; // skip
				
				float distSq = pow(x - pt[0], 2) + pow(y - pt[1], 2) + pow(z - pt[2], 2);
				
				if (distSq < minDistSq) {
					minDistSq = distSq;
					minIdx = i;
				}
			}
			
			return flat_g + minIdx * 4;
		}
		
		__global const float* closestPoint(__global const float *flat_g, float *pt) {
			
			__global const float *p = flat_g;
			int ax = (int) p[0];
			float split = p[1];
			int n = (int) p[2];
			int childCnt = (int) p[3];
			int nL = (int) p[4];
			int nR = (int) p[5];
			__global const float *current_best = 0;
			__global const float *new_best = 0;
			float best_dist;
			float dist;
			int crossBorder = 0;
			
			p += 6;
			
			for (;;) {
				if (pt[ax] <= split || crossBorder == -1) { // left
					if (nL >= 0) {
						new_best = closestPoint2(p, nL, pt);
					} else {
						new_best = closestPoint(p, pt);
					}
					
					dist = sqrt(pow(new_best[0] - pt[0], 2) +
						pow(new_best[1] - pt[1], 2) +
						pow(new_best[2] - pt[2], 2));
						
					if (current_best == 0 || dist < best_dist) {
						current_best = new_best;
						best_dist = dist;
					}
						
					if (dist + pt[ax] > split && crossBorder == 0) {
						crossBorder = 1;
					} else {
						return current_best;
					}
				}
				
				if (pt[ax] > split || crossBorder == 1) { // right
					if (nL >= 0) {
						p += nL * 4;
					} else {
						nL = (int) p[2];
						int childCntL = (int) p[3];
						
						p += childCntL * 6 + nL * 4;
					}
					
					if (nR >= 0) {
						new_best = closestPoint2(p, nR, pt);
					} else {
						new_best = closestPoint(p, pt);
					}
					
					dist = sqrt(pow(new_best[0] - pt[0], 2) +
						pow(new_best[1] - pt[1], 2) +
						pow(new_best[2] - pt[2], 2));
						
					if (current_best == 0 || dist < best_dist) {
						current_best = new_best;
						best_dist = dist;
					}
						
					if (pt[ax] - dist <= split && crossBorder == 0) {
						crossBorder = -1;
						p = flat_g + 6;
					} else {
						return current_best;
					}
				}
			}
		}
	
		__kernel void nn3d(__global const float *flat_g,
			__global const float *minima_g,
			__global const float *maxima_g,
			__global const int *res,
			__global float *out_g,
			__global float *accum_g,
			__global float *cnt_g,
			int ofs_x,
			int ofs_y) {
			
			float span[3] = {maxima_g[0] - minima_g[0],
				maxima_g[1] - minima_g[1],
				maxima_g[2] - minima_g[2]};
			
			int x = ofs_x + get_global_id(0);
			int y = ofs_y + get_global_id(1);
			int z = get_global_id(2);
			
			int ofs = (z * res[1] + y) * res[0] + x;
			// *(out_g + ofs) = (float)(x + y + z);
			
			// return;
			
			float pt[3];
			pt[0] = minima_g[0] + (maxima_g[0] - minima_g[0]) * x / ((float) res[0] - 1.0f);
			pt[1] = minima_g[1] + (maxima_g[1] - minima_g[1]) * y / ((float) res[1] - 1.0f);
			pt[2] = minima_g[2] + (maxima_g[2] - minima_g[2]) * z / ((float) res[2] - 1.0f);
			
			__global const float *closest = closestPoint(flat_g, pt);
			
			// *(out_g + ofs) = closest[3];
			
			// return;
			
			float radiusSq = pow(closest[0] - pt[0], 2) +
				pow(closest[1] - pt[1], 2) +
				pow(closest[2] - pt[2], 2);
			float radius = sqrt(radiusSq);
			float val = closest[3];
			
			int radius_x = ceil(radius * res[0] / (maxima_g[0] - minima_g[0]));
			int radius_y = ceil(radius * res[1] / (maxima_g[1] - minima_g[1]));
			int radius_z = ceil(radius * res[2] / (maxima_g[2] - minima_g[2]));
			
			for (int x2 = max(0, x - radius_x); x2 <= min(res[0] - 1, x + radius_x); x2++) {
				float x3 = minima_g[0] + span[0] * x2 / ((float) res[0] - 1.0f);
				float dxSq = pow(x3 - pt[0], 2);
				
				for (int y2 = max(0, y - radius_y); y2 <= min(res[1] - 1, y + radius_y); y2++) {
					float y3 = minima_g[1] + span[1] * y2 / ((float) res[1] - 1.0f);
					float dySq = pow(y3 - pt[1], 2);
					
					for (int z2 = max(0, z - radius_z); z2 <= min(res[2] - 1, z + radius_z); z2++) {
						float z3 = minima_g[2] + span[2] * z2 / ((float) res[2] - 1.0f);
						float dzSq = pow(z3 - pt[2], 2);
						
						float distSq = dxSq + dySq + dzSq;
						
						if (distSq <= radiusSq) {
							int ofs = (z2 * res[1] + y2) * res[0] + x2;
							
							atomic_add_global(accum_g + ofs, val);
							atomic_add_global(cnt_g + ofs, 1.0f);
						}
					}
				}
			}
		}
		""").build()
		
	for x in xrange(res[0]):
		for y in xrange(res[1]):
			print 'x:', x, 'y:', y
		
			ev = prg.nn3d(queue, [1, 1, out.shape[2]], None, flat_g, minima_g, maxima_g,
				res_g, out_g, accum_g, cnt_g, np.int32(x), np.int32(y))
			print 'ev:', ev
			# cl.enqueue_barrier(queue, wait_for=[ev])
			# ev.wait()
			cl.wait_for_events([ev])
			
		
	cl.enqueue_copy(queue, accum, accum_g)
	cl.enqueue_copy(queue, cnt, cnt_g)
	return (accum, cnt)

	# return out