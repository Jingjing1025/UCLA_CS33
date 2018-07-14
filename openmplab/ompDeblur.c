//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Jingjing Nie
 * UCLA ID: 304567417
 * Email id: niejingjing@g.ucla.edu
 * Input : New files
*/
 
/*
 Test result:
 12.131445 on lnxsrv08
 21.181685 on lnxsrv01
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

int OMP_Index(int x, int y, int z)
{
    return ((((z << 7) + y) << 7) + x);
}
#define Index(x, y, z) OMP_Index(x, y, z)

double OMP_SQR(double x)
{
    return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
    xMax = xM;
    yMax = yM;
    zMax = zM;
    assert(OMP_conv = (double*)malloc(sizeof(double) << 21));
    assert(OMP_g = (double*)malloc(sizeof(double) << 21));
}
void OMP_Finish()
{
    free(OMP_conv);
    free(OMP_g);
}

#define numThreads 16

void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
    double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
    double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
    int x, y, z, step, val;
    double boundryScale = 1.0 / (1.0 - nu);
    double postScale = pow(nu / lambda, (double)(3 * stepCount));
    

    for(step = 0; step < stepCount; step++)
    {
#pragma omp parallel for num_threads(numThreads) private(x,y,z)

        for(z = 0; z < zMax; z++)
        {
            for(y = 0; y < yMax; y++)
            {
                val = ((z << 7) + y) << 7;

                u[val] *= boundryScale;
                
                for(x = 1; x < xMax-1; x+=2)
                {
                    u[val + x] += u[val + x - 1] * nu;
                    u[val + x + 1] += u[val + x] * nu;
                }
                for (; x < xMax; x++)
                    u[val + x] += u[val + x - 1] * nu;
                
                
                u[val] *= boundryScale;
                
                for(x = xMax - 2; x >= 0; x--)
                {
                    u[val + x] += u[val + x + 1] * nu;
                }
            }
            
            for(x = 0; x < xMax; x++)
            {
                u[(z << 14) + x] *= boundryScale;
            }
            
            for(y = 1; y < yMax; y++)
            {
                val = ((z << 7) + y) << 7;
                
                for(x = 0; x < xMax; x++)
                {
                    u[val + x] += u[val + x - xMax] * nu;
                }
            }

            for(x = 0; x < xMax; x++)
            {
                u[Index(x, yMax - 1, z)] *= boundryScale;
            }

            for(y = yMax - 2; y >= 0; y--)
            {
                val = ((z << 7) + y) << 7;
                
                for(x = 0; x < xMax-1; x+=2)
                {
                    u[val + x] += u[val + xMax + x] * nu;
                    u[val + x + 1] += u[val + xMax + x + 1] * nu;
                }
                for (; x < xMax; x++)
                    u[val + x] += u[val + xMax + x] * nu;
            }
        }
        
#pragma omp parallel for num_threads(numThreads) private(x,y,z)
        
        for(y = 0; y < yMax; y++)
        {
            for(x = 0; x < xMax; x++)
            {
                val = (y << 7) + x;

                u[val] *= boundryScale;
            }
            
            for(z = 1; z < zMax; z++)
            {
                val = ((z << 7) + y) << 7;

                for(x = 0; x < xMax; x++)
                {
                    u[val + x] = u[Index(x, y, z - 1)] * nu;
                }
            }
            
            for(x = 0; x < xMax; x++)
            {
                u[Index(x, y, zMax - 1)] *= boundryScale;
            }
            
            for(z = zMax - 2; z >= 0; z--)
            {
                for(x = 0; x < xMax-1; x+=2)
                {
                    u[Index(x, y, z)] += u[Index(x, y, z + 1)] * nu;
                    u[Index(x+1, y, z)] += u[Index(x+1, y, z + 1)] * nu;
                }
            }
            
        }
    }
    
#pragma omp parallel for num_threads(numThreads) private(x,y,z)

    for(z = 0; z < zMax; z++)
    {
        for(y = 0; y < yMax; y++)
        {
            val = ((z << 7) + y) << 7;
            
            for(x = 0; x < xMax-1; x+=2)
            {
                u[val + x] *= postScale;
                u[val + x + 1] *= postScale;
            }
            for (; x < xMax; x++)
                u[val + x] *= postScale;
        }
    }
}
void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
    double epsilon = 1.0e-7;
    double sigma2 = SQR(sigma);
    int x, y, z, iteration;
    int converged = 0;
    int lastConverged = 0;
    int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
    double* conv = OMP_conv;
    double* g = OMP_g;
    
    for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
    {
#pragma omp parallel for num_threads(numThreads) private(x,y,z)
        for(z = 1; z < zMax - 1; z++)
        {
            for(y = 1; y < yMax - 1; y++)
            {
                for(x = 1; x < xMax - 1; x++)
                {
                    int ind = Index(x,y,z);
                    double indVal = u[ind];
                    g[ind] = 1.0 / sqrt(epsilon +
                                        SQR(indVal - u[ind + 1]) +
                                        SQR(indVal - u[ind - 1]) +
                                        SQR(indVal - u[ind + xMax]) +
                                        SQR(indVal - u[ind - xMax]) +
                                        SQR(indVal - u[Index(x, y, z + 1)]) +
                                        SQR(indVal - u[Index(x, y, z - 1)]));
                }
            }
        }
        memcpy(conv, u, sizeof(double) << 21);
        OMP_GaussianBlur(conv, Ksigma, 3);
        
#pragma omp parallel for num_threads(numThreads) private(x,y,z)

        for(z = 0; z < zMax; z++)
        {
            for(y = 0; y < yMax; y++)
            {
                for(x = 0; x < xMax; x++)
                {
                    int ind = Index(x,y,z);
                    double r = conv[ind] * f[ind] / sigma2;
                    r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
                    conv[ind] -= f[ind] * r;
                }
            }
        }
        OMP_GaussianBlur(conv, Ksigma, 3);
        converged = 0;
        for(z = 1; z < zMax - 1; z++)
        {
            for(y = 1; y < yMax - 1; y++)
            {
                for(x = 1; x < xMax - 1; x++)
                {
                    int ind = Index(x, y, z);
                    double oldVal = u[ind];
                    double newVal = (oldVal + dt * (
                                                    u[ind - 1] * g[ind - 1] +
                                                    u[ind + 1] * g[ind + 1] +
                                                    u[ind - xMax] * g[ind - xMax] +
                                                    u[ind + xMax] * g[ind + xMax] +
                                                    u[Index(x, y, z - 1)] * g[Index(x, y, z - 1)] +
                                                    u[Index(x, y, z + 1)] * g[Index(x, y, z + 1)] - gamma * conv[ind])) /
                    (1.0 + dt * (g[ind + 1] + g[ind - 1] + g[ind + xMax] + g[ind - xMax] + g[Index(x, y, z + 1)] + g[Index(x, y, z - 1)]));
                    if(fabs(oldVal - newVal) < epsilon)
                    {
                        converged++;
                    }
                    u[ind] = newVal;
                }
            }
        }
        if(converged > lastConverged)
        {
            printf("%d pixels have converged on iteration %d\n", converged, iteration);
            lastConverged = converged;
        }
    }
}

