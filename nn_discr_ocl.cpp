#if 0
g++ -framework OpenCL -O3 -shared -o libnn_discr_ocl.so nn_discr_ocl.cpp
cython pynn_discr_ocl.pyx
gcc -shared pynn_discr_ocl.c -fPIC -L/opt/local/lib -lpython2.7 -L`pwd` -lnn_discr_ocl -o pynn_discr_ocl.so \
    -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 \
    -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include
exit 0
#endif

//
// Copyright (C) Stanislaw Adaszewski, 2016
// http://algoholic.eu
//

#include <OpenCL/cl.h>
#include <stdio.h>
#include <math.h>

extern "C" {

void nn_discrete(double ax, double bx, double ay, double by, double az, double bz,
                 double *x, double *y, double *z,
                 double *v1, double *v2, double *v3,
                 int n, int kx, int ky, int kz,
                 double *out1, double *out2, double *out3) {
    for (int i = 0; i < n; i++) {
        printf("%.2f %.2f %.2f %.2f %.2f %.2f\n", x[i], y[i], z[i], v1[i], v2[i], v3[i]);
    }
    int n1 = kx * ky * kz;
    int *voro_pt = new int[n1];
    double *voro_dist = new double[n1];
    double inf = 1.0 / 0.0;
    double sx = (bx - ax) / (kx - 1);
    double sy = (by - ay) / (ky - 1);
    double sz = (bz - az) / (kz - 1);
    printf("sx: %.2f, sy: %.2f, sz: %.2f\n", sx, sy, sz);
    for (int i = 0; i < n1; i++) voro_dist[i] = inf;
    printf("Discrete Voronoi...\n");
    int idx = 0;
    for (int i = 0; i < kx; i++) for (int k = 0; k < ky; k++) for (int m = 0; m < kz; m++, idx++) {
        double x1 = ax + i * sx, y1 = ay + k * sy, z1 = az + m * sz;
        for (int w = 0; w < n; w++) {
            double dist = pow((x[w] - x1) / sx, 2) + pow((y[w] - y1) / sy, 2) + pow((z[w] - z1) / sz, 2);
            if (dist < voro_dist[idx]) {
                voro_dist[idx] = dist;
                voro_pt[idx] = w;
            }
        }
    }
    printf("Discrete NN...\n");
    idx = 0;
    // double extra_r = 0; // 2*sqrt(3); // sx * sx + sy * sy + sz * sz);
    int *cnt1 = new int[n1]();
    //int *cnt2 = new int[n1];
    //int *cnt3 = new int[n1];
    //for (int i = 0; i < n1; i++) cnt1[i] = cnt2[i] = cnt3[i] = 0;
    for (int i = 0; i < kx; i++) { printf("i: %d\n", i); for (int k = 0; k < ky; k++) for (int m = 0; m < kz; m++, idx++) {
        double r2 = voro_dist[idx];
        int pt = voro_pt[idx];
        int r = (int) ceil(sqrt(r2));
        // if (r % 2 == 1) r -= 1;
        double ax = fmax(0, i - r), bx = fmin(kx - 1, i + r);
        double ay = fmax(0, k - r), by = fmin(ky - 1, k + r);
        double az = fmax(0, m - r), bz = fmin(kz - 1, m + r);
        // printf("i: %d, k: %d, m: %d, r: %.2f (%d), pt: %d, ax: %.2f, bx: %.2f, ay: %.2f, by: %.2f, az: %.2f, bz: %.2f\n", i, k, m, sqrt(r2), r, pt, ax, bx, ay, by, az, bz);
        for (int i1 = ax; i1 <= bx; i1++) for (int k1 = ay; k1 <= by; k1++) for (int m1 = az; m1 <= bz; m1++) {
            double dist = sqrt(pow(i1 - i, 2) + pow(k1 - k, 2) + pow(m1 - m, 2));
            double r = sqrt(r2);
            int idx1 = (i1 * ky + k1) * kz + m1;
            if (dist <= r) {
                // double V = 1;//M_PI * pow(r + extra_r - dist, 2) * (pow(dist, 2) + 2 * dist * extra_r - 3 * pow(extra_r, 2) + 2 * dist * r + 6 * r * extra_r - 3 * r * r) / (12 * dist);
                //printf("V: %.2f\n", V);
                //V /= 
                // if (V > 0) {
                out1[idx1] += v1[pt];
                out2[idx1] += v2[pt];
                out3[idx1] += v3[pt];
                cnt1[idx1]++;
                // }
                //cnt2[idx1]++;
                //cnt3[idx1]++;
            }
        }
    } }
    for (int i = 0; i < n1; i++) {
        out1[i] /= cnt1[i];
        out2[i] /= cnt1[i];
        out3[i] /= cnt1[i];
        // out1[i] = voro_pt[i];
        // out2[i] = voro_dist[i];
        // out3[i] = cnt1[i];
    }
    delete[] cnt1;
    //delete[] cnt2;
    //delete[] cnt3;
    delete[] voro_pt;
    delete[] voro_dist;
}

}

int main() {
    double x[] = {.2, .5, .8};
    double y[] = {.2, .5, .8};
    double z[] = {.2, .5, .8};
    double v1[] = {1, 2, 3};
    double v2[] = {4, 5, 6};
    double v3[] = {7, 8, 9};
    int kx = 69, ky = 177, kz = 25;
    int n1 = kx * ky * kz;
    double *out1 = new double[n1];
    double *out2 = new double[n1];
    double *out3 = new double[n1];
    nn_discrete(0, 1, 0, 1, 0, 1, x, y, z, v1, v2, v3, 3, kx, ky, kz, out1, out2, out3);
}
