#
# Discrete 3D Natural Neighbor Approximation for OpenCL
# 
# Copyright (C) Stanislaw Adaszewski, 2016
# http://algoholic.eu
#


import numpy as np
import pyopencl as cl
import time


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
		

def closestPoint(node, pt):
	if isinstance(node, KDTreeNode):
		# print 'node ax:', node.ax, 'split:', node.split
		if pt[node.ax] <= node.split:
			cur_best = closestPoint(node.L, pt)
			cur_dist = np.linalg.norm(cur_best[:3] - pt)
			if len(cur_best) == 0 or pt[node.ax] + cur_dist > node.split:
				new_best = closestPoint(node.R, pt)
				new_dist = np.linalg.norm(new_best[:3] - pt)
				if len(new_best) > 0 and new_dist < cur_dist:
					return new_best
				else:
					return cur_best
			return cur_best
		else:
			cur_best = closestPoint(node.R, pt)
			cur_dist = np.linalg.norm(cur_best[:3] - pt)
			if len(cur_best) == 0 or pt[node.ax] - cur_dist <= node.split:
				new_best = closestPoint(node.L, pt)
				new_dist = np.linalg.norm(new_best[:3] - pt)
				if len(new_best) > 0 and new_dist < cur_dist:
					return new_best
				else:
					return cur_best
			return cur_best
	
	else:
		# print 'points...'
		if len(node) == 0:
			return []
		tmp = np.linalg.norm(node[:, :3] - pt, axis=1)
		# print 'tmp:', tmp
		return node[np.argsort(tmp)[0], :]


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


def nn3d(pts, values,
	minima=np.zeros((3,), dtype=np.float32),
	maxima=np.ones((3,), dtype=np.float32),
	res=np.ones((3,), dtype=np.int32) * 32, ):
	
	ctx = cl.create_some_context()
	queue = cl.CommandQueue(ctx)
	mf = cl.mem_flags
	
	
	minima_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = minima)
	span = maxima - minima;
	span_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = span)
	res_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = res)
	
	val = np.zeros(res, dtype=np.float32)
	val_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = val)
	
	radius = np.zeros(res, dtype=np.float32)
	radius_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = radius)
	
	pts2 = np.hstack((pts, np.reshape(values, (len(values),1))))
	print 'pts2:', pts2
	root = buildKDTree(pts2)
	
	flat = flattenKDTree(root)
	flat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = flat)
	
	accum = np.zeros(res, dtype=np.float32)
	cnt = np.zeros(res, dtype=np.float32)
	
	accum_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=accum)
	cnt_g = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=cnt)
	
	prg = cl.Program(ctx, """
		void atomic_add_global(volatile __global float *source, const float operand) {
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
	
		__kernel void fillClosestPtIds(__global const float *flat_g,
			__global const float *minima_g,
			__global const float *span_g,
			__global const int *res,
			__global float *val_g,
			__global float *radius_g) {
			
			
			int x = get_global_id(0);
			int y = get_global_id(1);
			int z = get_global_id(2);
			
			int ofs = (z * res[1] + y) * res[0] + x;
			
			float pt[3];
			pt[0] = minima_g[0] + span_g[0] * x / ((float) res[0] - 1.0f);
			pt[1] = minima_g[1] + span_g[1] * y / ((float) res[1] - 1.0f);
			pt[2] = minima_g[2] + span_g[2] * z / ((float) res[2] - 1.0f);
			
			__global const float *closest = closestPoint(flat_g, pt);
			
			float radius = sqrt(pow(pt[0] - closest[0], 2) +
				pow(pt[1] - closest[1], 2) +
				pow(pt[2] - closest[2], 2));
			
			*(val_g + ofs) = closest[3];
			*(radius_g + ofs) = radius;
		}
	
		__kernel void drawSphere(__global const float *minima_g,
			__global const float *span_g,
			__global const int *res,
			int ofs_x, int ofs_y, int ofs_z,
			float cx, float cy, float cz,
			float radiusSq,
			float val,
			__global float *accum_g,
			__global float *cnt_g) {
			
			int ix = ofs_x + get_global_id(0);
			int iy = ofs_y + get_global_id(1);
			int iz = ofs_z + get_global_id(2);
			
			if (ix < 0 || ix >= res[0]) return;
			if (iy < 0 || iy >= res[1]) return;
			if (iz < 0 || iz >= res[2]) return;
			
			float x = minima_g[0] + span_g[0] * ix / res[0];
			float y = minima_g[1] + span_g[1] * iy / res[1];
			float z = minima_g[2] + span_g[2] * iz / res[2];
			
			float distSq = pow(x - cx, 2) +
				pow(y - cy, 2) +
				pow(z - cz, 2);
			
			if (distSq <= radiusSq) {
				int ofs = (iz * res[1] + iy) * res[0] + ix;
				// *(accum_g + ofs) += val;
				// *(cnt_g + ofs) += 1.0f;
				atomic_add_global(accum_g + ofs, val);
				atomic_add_global(cnt_g + ofs, 1.0f);
			}
		}
		""").build()
		
	prg.fillClosestPtIds(queue, radius.shape, None,
		flat_g, minima_g, span_g, res_g,
		val_g, radius_g);
		
	cl.enqueue_copy(queue, radius, radius_g)
	cl.enqueue_copy(queue, val, val_g)
	
	print 'Radius and value map done'
		
	kd_overhead = 0.0
	t0 = time.time()
	for ix in xrange(res[0]):
		print 'ix:', ix
		for iy in xrange(res[1]):
			# print 'iy:', iy
			for iz in xrange(res[2]):
				x = minima[0] + span[0] * ix / (res[0] - 1)
				y = minima[1] + span[1] * iy / (res[1] - 1)
				z = minima[2] + span[2] * iz / (res[2] - 1)
				
				v = val[ix, iy, iz]
				r = radius[ix, iy, iz]
				rx = int(np.ceil(r / span[0] * res[0]))
				ry = int(np.ceil(r / span[1] * res[1]))
				rz = int(np.ceil(r / span[2] * res[2]))
				radiusSq = r ** 2
				shape = [2*rx, 2*ry, 2*rz]
				# print 'shape:', shape
				ev = prg.drawSphere(queue, shape, None,
					minima_g, span_g, res_g,
					np.int32(ix - rx), np.int32(iy - ry), np.int32(iz - rz),
					np.float32(x), np.float32(y), np.float32(z),
					np.float32(radiusSq), np.float32(v),
					accum_g, cnt_g);
				# cl.wait_for_events([ev])
	
	print 'Time spent in KD-Tree search:', kd_overhead, 's'

	cl.enqueue_copy(queue, accum, accum_g)
	cl.enqueue_copy(queue, cnt, cnt_g)
	
	print 'Total time:', time.time() - t0, 's'
	
	return (accum, cnt)

	# return out