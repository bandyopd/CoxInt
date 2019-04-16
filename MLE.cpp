//
//  Cox regression model with interval censoring
//
//  Created by Piet Groeneboom on 26-03-2019.
//  Copyright (c) 2018 Piet. All rights reserved.
//

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <time.h>
#include <fstream>
#include <string.h>
#include <Rcpp.h>

#define SQR(x) ((x)*(x))

using namespace std;
using namespace Rcpp;

#define SQR(x) ((x)*(x))
#define SWAP(a,b) itemp=(a);(a)=(b);(b)=itemp;
#define M 7
#define NSTACK 50

typedef struct
{
    int index;
    double v;
    int second;
}
data_object;

int     n,m1,nn,nreg,first,last,*second,*ind,iteration,**freq;
double  *beta0,**zz,**zz2,*Lambda,*Lambda_new,*Lam,*predictor,**aa,**bb;
double  *cumw,*cs,*tt,*uu,*vv,*y,*support;
double  *F,*Lambda1,*d,*g,*w,*Fn,phi;

int     compare(const void *a, const void *b);
void    sortcens(int n, int *nn, double vv[]);
void    rank2(int n, int indx[], int irank[]);
void    indexx(int n, double arr[], int indx[]);

double  f_alpha(double alpha, double F[], double F_new[]);
double  f_alpha_prime(double alpha, double F[], double F_new[]);
double  golden(double a1, double b1, double F[], double F_new[], double (*f)(double,double*,double*));
void    Compute_a(int nn, int m1, double F[], double beta[]);


//sort.cpp
void swap(double *x,double *y);
void rank(int n, int indx[], int irank[]);
void indexx(int n, double arr[], int indx[]);
void sortcens(int n, int *nn, int m1, double vv[]);
void sort_vv(int n, int nn, double vv[]);


//icm.cpp
double  icm(int m1, double beta[]);
int     FirstIndex(int n, int second[]);
int     LastIndex(int n, int second[]);
void    isreg(double d[], double g[], double F1[], double gradb[], double w[]);
void    cumsum(double b[], double nabla[], double w[]);
void    convexminorant(double g[], double w[]);
int     fenchelviol(double b[], double nabla[], double tol, double *inprod, double *partsum);
void    initializeEM(int n, double p[], double b[]);
void    initializeICM(int n, double b[], double d[],double g[], double w[]);
void    gradient(int m1, double nabla[], double b[], int second[], double beta[]);
void    weights(double nabla[], double w[]);
void    compute_Lambda1(double Lambda[]);
int     ComputeNumberofJumps(int nn,double F[]);
void    detLamOnJumppoints(int nn,double y[], double b[], double Lam[], double support[]);

int     hooke(int m, double startpt[], double endpt[], double rho, double eps,
              int itermax, double f(int m, double alpha[]));

// [[Rcpp::export]]

List ComputeMLE(NumericMatrix X, NumericMatrix Y, NumericVector beta0, int n1, int m)
{
    int         i,j,iter,nIterations;
    double      rho,eps,*beta,*beta_init,*beta0_copy;
    
    n=(int)n1;
    
    // m1 is the dimension
    m1= (int)m;
    
    rho=0.5;
    eps=1.0e-7;
    nIterations=1000;
    
    beta0_copy  = new double[m1+1];
    beta        = new double[m1+1];
    beta_init   = new double[m1+1];
    tt          = new double[n+1];
    uu          = new double[n+1];
    vv          = new double[2*n+2];
    F           = new double[2*n+2];
    Fn          = new double[2*n+2];
    Lambda_new  = new double[2*n+2];
    Lambda      = new double[2*n+2];
    Lambda1      = new double[2*n+2];
    Lam         = new double[2*n+2];
    predictor   = new double[2*n+2];
    cumw        = new double[2*n+2];
    cs          = new double[2*n+2];
    y           = new double[2*n+2];
    support     = new double[2*n+2];
    d           = new double[2*n+2];
    g           = new double[2*n+2];
    w           = new double[2*n+2];
    
    aa = new double *[2*n+2];
    for (i=0;i<2*n+2;i++)
        aa[i] = new double [m1+2];
    
    bb = new double *[2*n+2];
    for (i=0;i<2*n+2;i++)
        bb[i] = new double [m1+2];
    
    zz = new double *[2*n+2];
    for (i=0;i<2*n+2;i++)
        zz[i] = new double [m1+2];
    
    zz2 = new double *[2*n+2];
    for (i=0;i<2*n+2;i++)
        zz2[i] = new double [m1+2];
    
    ind=new int[2*n+2];
    second = new int[2*n+2];
    
    for (i=0;i<m1;i++)
        beta0_copy[i]=(double)beta0(i);
    
    for (i=0;i<n;i++)
    {
        for (j=0;j<m1;j++)
            zz[i+1][j+1]=X(i,j);
    }
        
    for (i=0;i<n;i++)
    {
        tt[i+1]=(double)Y(i,0);
        uu[i+1]=(double)Y(i,1);
        ind[i+1]=(int)Y(i,2);
    }
    
    sortcens(n,&nn,m1,vv);
    
    first = FirstIndex(nn,second);
    last = LastIndex(nn,second);
    
    for (i=0;i<m1;i++)
        beta_init[i]=0;
    
    iter=hooke(m1,beta_init,beta,rho,eps,nIterations,icm);
    
    //printf("%5d",iter+1);
    
    //for (i=0;i<m1;i++)
        //printf("%15.10f",beta[i]);
    
    //printf("\n");
    
    NumericMatrix out0 = NumericMatrix(n,3);
    
    for (i=0;i<n;i++)
    {
        out0(i,0)=tt[i+1];
        out0(i,1)=uu[i+1];
        out0(i,2)=ind[i+1];
    }
    
    NumericVector out1 = NumericVector(m);
    
    // computation of beta
    
    for (i=0;i<m1;i++)
        out1(i)=beta[i];
    
    NumericVector out2 = NumericMatrix(last-first+1,2);
    for (i=0;i<=last-first;i++)
    {
        out2(i,0)=vv[i+first];
        out2(i,1)=Lambda[i+first];
    }
    
    for (i=first;i<=last;i++)
        F[i]=1-exp(-Lambda[i]);
    
    NumericVector out3 = NumericMatrix(last-first+1,2);
    for (i=0;i<=last-first;i++)
    {
        out3(i,0)=vv[i+first];
        out3(i,1)=F[i+first];
    }
    
    
    // make the list for the output, containing beta and the estimate of the baseline cumulative hazard
    
    List out = List::create(Rcpp::Named("data")=out0,Rcpp::Named("beta")=out1,Rcpp::Named("Lambda")=out2,Rcpp::Named("F")=out3);
    
    
    
    // free memory
    
    delete[] beta0_copy; delete[] beta; delete[] beta_init; delete[] tt; delete[] uu; delete[] vv;
    delete[] F; delete[] Fn; delete[] Lambda; delete[] Lambda_new; delete[] Lambda1; delete[] Lam;
    delete[] predictor; delete[] cumw; delete[] cs; delete[] y; delete[] support;
    delete[] d; delete[] g; delete[] w;
    delete[] ind; delete[] second;
    
    for (i=0;i<2*n+2;i++)
        delete[] aa[i];
    delete[] aa;
    
    for (i=0;i<2*n+2;i++)
        delete[] bb[i];
    delete[] bb;
    
    for (i=0;i<2*n+2;i++)
        delete[] zz[i];
    delete[] zz;
    
    for (i=0;i<2*n+2;i++)
        delete[] zz2[i];
    delete[] zz2;
  
    
    return out;
}

double    icm(int m1, double beta[])
{
    int         i,j,iterations=1000;
    double      inprod, partialsum;
    double       *gradb;
    double       alpha,phib;
    double       eta=10e-5;
    int          dummy;
    
    gradb = new double[nn+2];
    
    Lambda[0]=0;
    g[0]=0;
    
    cumw[0]=0;
    cs[0]=0;
    
    initializeICM(nn,Lambda,d,g,w);
    
    // Compute nabla and nreg
    
    gradient(m1,gradb,Lambda,second,beta);
    
    phib= f_alpha(0.0,Lambda,Lambda);
    
    compute_Lambda1(Lambda);
    
    dummy = fenchelviol(Lambda1,gradb,eta,&inprod,&partialsum);
    
    j=1;
    
    while ((j<=iterations) && fenchelviol(Lambda1,gradb,eta,&inprod,&partialsum))
    {
        j++;
        
        //printf("%5d     %5d    %15.10f  %15.10f  %15.10f    %15.10f  %15.10f  %15.10f\n",j,nreg,inprod,partialsum,phib,beta[0],beta[1],beta[2]);
        
        isreg(d,g,Lambda1,gradb,w);
        
        if (f_alpha_prime(1.0,Lambda,g)<=0) alpha=1;
        else
            alpha=golden(0.1,1,Lambda,g,f_alpha);
        
        for (i=first;i<=last;i++)
            Lambda[i] += alpha*(g[i]-Lambda[i]);
        
        phib= f_alpha(0.0,Lambda,Lambda);
        
        phi=phib;
        
        compute_Lambda1(Lambda);
        
        // Compute nabla
        gradient(m1,gradb,Lambda,second,beta);
    }
    
    phib= f_alpha(0.0,Lambda,Lambda);
    
    iteration=j;
    
    //printf("%5d   %15.10f   %15.10f  %15.10f  %15.10f    %15.10f  %15.10f\n",j,phib,inprod,partialsum,beta[0],beta[1],beta[2]);
    
    delete[] gradb;
    
    return phib;
    
}


void initializeICM(int n, double Lambda[], double d[],double g[], double w[])
{
    int    i;
    
    for (i=0;i<first;i++)
    {
        Lambda[i] =0;
        g[i]=Lambda[i];
        d[i]=g[i];
        Lambda_new[i]=0;
    }
    
    for (i=first;i<=last;i++)
    {
        if (vv[i]>vv[i-1] && second[i]!=nn+1)
        {
            Lambda[i]=Lambda[i-1]+1.0/(last-first+1);
            Lambda_new[i]=Lambda[i];
            g[i]=d[i]=Lambda[i];
        }
        else
        {
            Lambda[i]=Lambda[i-1];
            Lambda_new[i]=Lambda[i];
            g[i]=d[i]=Lambda[i];
        }
    }
    
    for (i=last+1;i<=nn+1;i++)
    {
        g[i]=1.0e6;
        Lambda[i]=1.0e6;
        d[i]=g[i];
        Lambda_new[i]=Lambda[i];
    }
}

double  f_alpha(double alpha, double Lambda[], double Lambda_new[])
{
    int    i;
    double    a,b,sum;
    
    sum=0;
    
    for (i=first;i<=last;i++)
    {
        a = (1-alpha)*Lambda[i]+alpha*Lambda_new[i];
        
        if (second[i]==0)
            sum += log(1-exp(-a*predictor[i]));
        else
        {
            if (second[i]>=1 && second[i]<=nn)
            {
                if (second[i]>i)
                {
                    if (second[i]<=last)
                    {
                        b = (1-alpha)*Lambda[second[i]]+alpha*Lambda_new[second[i]];
                        sum += 0.5*log(exp(-a*predictor[i])-exp(-b*predictor[i]));
                    }
                    else
                        sum -= a*predictor[i];
                }
                else
                {
                    if (second[i]>=first)
                    {
                        b = (1-alpha)*Lambda[second[i]]+alpha*Lambda_new[second[i]];
                        sum += 0.5*log(1-exp(-a*predictor[i]));
                    }
                    else
                        sum += log(1-exp(-a*predictor[i]));
                }
            }
            else
            {
                if (second[i]==nn+1)
                    sum -= a*predictor[i];
            }
        }
    }
    
    return -sum/n;
}


double f_alpha_prime(double alpha, double Lambda[], double Lambda_new[])
{
    int i;
    double a,b,sum;
    
    sum=0;
    
    for (i=first;i<=last;i++)
    {
        a = (1-alpha)*Lambda[i]+alpha*Lambda_new[i];
        if (second[i]==0)
            sum += predictor[i]*(Lambda_new[i]-Lambda[i])*exp(-a*predictor[i])/(1-exp(-a*predictor[i]));
        else
        {
            if (second[i]>=1 && second[i]<=nn)
            {
                if (second[i]>i)
                {
                    if (second[i]<=last)
                    {
                        b = (1-alpha)*Lambda[second[i]]+alpha*Lambda_new[second[i]];
                        sum -= predictor[i]*(Lambda_new[i]-Lambda[i])*exp(-a*predictor[i])/(exp(-a*predictor[i])-exp(-b*predictor[i]));
                    }
                    else
                        sum -= predictor[i]*(Lambda_new[i]-Lambda[i]);
                    
                }
                else
                {
                    if (second[i]>=first)
                    {
                        b = (1-alpha)*Lambda[second[i]]+alpha*Lambda_new[second[i]];
                        sum += predictor[i]*(Lambda_new[i]-Lambda[i])*exp(-a*predictor[i])/(exp(-b*predictor[i])-exp(-a*predictor[i]));
                    }
                    else
                        sum += predictor[i]*(Lambda_new[i]-Lambda[i]);
                }
            }
            else
                sum -= predictor[i]*(Lambda_new[i]-Lambda[i]);
        }
    }
    return -sum/n;
}


double golden(double a1, double b1, double Lambda[], double Lambda_new[], double (*f)(double,double*,double*))
{
    double a,b,eps=1.0e-5;
    
    a=a1;
    b=b1;
    
    double k = (sqrt(5.0) - 1.0) / 2;
    double xL = b - k*(b - a);
    double xR = a + k*(b - a);
    
    while (b-a>eps)
    {
        if ((*f)(xL,Lambda,Lambda_new)<(*f)(xR,Lambda,Lambda_new))
        {
            b = xR;
            xR = xL;
            xL = b - k*(b - a);
        }
        else
        {
            a = xL;
            xL = xR;
            xR = a + k * (b - a);
        }
    }
    return (a+b)/2;
    
}

int FirstIndex(int n, int second[])
{
    int i=1;
    
    while (second[i]>i && i<n)
        i++;
    
    return i;
}

int    LastIndex(int n, int second[])
{
    int i=n;
    
    while (second[i]<i)
        i--;
    
    return i;
}


int fenchelviol(double b[], double nabla[], double tol, double *inprod, double *partsum)
{
    double    sum,sum2;
    int    i;
    int    fenchelvioltemp;
    
    fenchelvioltemp = 0;
    
    sum=0;
    sum2=0;
    
    for (i=first;i<=nreg;i++)
    {
        sum -= nabla[i];
        if (sum < sum2)
            sum2 = sum;
    }
    
    sum=0;
    for (i=first;i<=nreg;i++)
        sum += nabla[i]*b[i];
    
    *inprod = sum;
    *partsum = sum2;
    
    if ((fabs(sum) > tol) || (sum2 < -tol) ) fenchelvioltemp = 1;
    
    return fenchelvioltemp;
}

// compute Lambda on the reduced set of points

void compute_Lambda1(double Lambda[])
{
    int i,j;
    
    j=first-1;
    Lambda1[j]=0;
    
    for (i=first;i<=last;i++)
    {
        if (vv[i]>vv[i-1] && second[i]!=nn+1)
        {
            j++;
            Lambda1[j]=Lambda[i];
        }
    }
}

void gradient(int m1, double nabla[], double F[], int second[], double beta[])
{
    int i,j;
    
    Compute_a(nn,m1,Lambda,beta);
    
    for (i=first;i<=last;i++)
        nabla[i]=0;
    
    j=first-1;
    for (i=first;i<=last;i++)
    {
        if (vv[i]>vv[i-1] && second[i]!=nn+1)
        {
            j++;
            if (second[i]==0)
                nabla[j]= -aa[i][1];
            else
            {
                if (second[i]>i)
                    nabla[j]= -aa[i][2];
                else
                    nabla[j] = -aa[i][3];
            }
        }
        else
        {
            if (second[i]==0)
                nabla[j] -= aa[i][1];
            else
            {
                if (second[i]==nn+1)
                    nabla[j] -= aa[i][4];
                else
                {
                    if (second[i]>i)
                        nabla[j] -= aa[i][2];
                    else
                        nabla[j] -= aa[i][3];
                }
            }
        }
    }
    nreg=j;
}

void     weights(double nabla[], double w[])
{
    int    i,j;
    double tol=1.0e-8;
    
    for (i=first;i<=last;i++)
        w[i]=0;
    
    j=first-1;
    
    for (i=first;i<=last;i++)
    {
        if (vv[i]>vv[i-1] && second[i]!=nn+1)
        {
            j++;
            if (second[i]==0)
                w[j] = bb[i][1];
            else
            {
                if (second[i]<i)
                    w[j] = bb[i][3];
                else
                {
                    if (second[i]<=nn)
                        w[j] = bb[i][2];
                }
            }
        }
        else
        {
            if (second[i]==0)
                w[j] += bb[i][1];
            else
            {
                if (second[i]<i)
                    w[j] += bb[i][3];
                else
                {
                    if (second[i]<=nn)
                        w[j] += bb[i][2];
                }
            }
        }
    }
    
    for (i=first;i<=nreg;i++)
        w[i]=fmax(fmin(w[i],100),tol);
}


void cumsum(double Lambda1[], double nabla[], double w[])
{
    int    j;
    
    cs[first-1]=0;
    
    for (j=first;j<=nreg;j++)
        cs[j] = cs[j-1]+Lambda1[j]*w[j]-nabla[j];
}


void isreg(double d[], double g[], double Lambda1[], double gradb[], double w[])
{
    int i,j;
    
    weights(gradb,w);
    cumsum(Lambda1,gradb,w);
    convexminorant(d,w);
    
    for (i=first;i<=nreg;i++)
    {
        if (d[i]<0)
            d[i] = 0;
    }
    
    j=first-1;
    for (i=first;i<=last;i++)
    {
        if (vv[i]>vv[i-1] && second[i]!=nn+1)
        {
            j++;
            g[i]=d[j];
        }
        else
            g[i]=d[j];
    }
    
}

void convexminorant(double g[], double w[])
{
    int    i, j, m;
    
    cs[first-1] = 0;
    cumw[first-1] = 0;
    
    for (i = first;i<=nreg;i++)
        cumw[i] = cumw[i-1] + w[i];
    // vector of cumulative weights is constructed
    
    g[first] = cs[first]/w[first];
    for (i=first+1;i<=nreg;i++)
    {
        g[i] = (cs[i]-cs[i-1])/w[i];
        if (g[i-1]>g[i])
        {
            j = i;
            while ((g[j-1] > g[i]) && (j>first))
            {
                j=j-1;
                g[i] = (cs[i]-cs[j-1])/(cumw[i]-cumw[j-1]);
                for (m=j;m<i;m++)    g[m] = g[i];
            }
        }
    }
}
int  ComputeNumberofJumps(int nn,double Lambda[])
{
    int i,count;
    
    count=0;
    
    Lambda[first-1]=0;
    for (i=first;i<=last;i++)
    {
        if (Lambda[i]>Lambda[i-1])
            count++;
    }
    
    if ((Lambda[last+1]>Lambda[last]) && (last<nn))
        count++;
    
    return count;
}

void detLamOnJumppoints(int nn, double b[], double Lam[], double gridofjumps[])
{
    int i,j;
    
    j=0;
    
    b[first-1]=0;
    Lam[0]=0;
    
    for (i=first;i<=last;i++)
    {
        if (b[i]>b[i-1])
        {
            j=j+1;
            Lam[j]=b[i];
            gridofjumps[j]=vv[i];
        }
        
    }
    
    if ((b[last+1]>b[last]) && (last<nn))
    {
        j=j+1;
        Lam[j]=1;
        gridofjumps[j]=vv[last+1];
    }
}

void Compute_a(int nn, int m1, double F[], double beta[])
{
    int    i,j;
    double    sum;
    
    for (i=1;i<=nn;i++)
    {
        for (j=1;j<=m1+1;j++)
            aa[i][j]=bb[i][j]=0;
        
        //note that the parameters are beta[0] to beta[m1-1] by the zero offset for the parameters in Hooke-Jeeves:
        
        sum=0;
        for(j=1;j<=m1;j++)
            sum += beta[j-1]*zz[i][j];
        
        predictor[i] = exp(sum);
    }
    
    
    for (i=first;i<=last;i++)
    {
        if (second[i]==0)
            aa[i][1] = predictor[i]*exp(-Lambda[i]*predictor[i])/(1-exp(-Lambda[i]*predictor[i]));
        else
        {
            if (second[i]>i)
            {
                if (second[i]<=last)
                    aa[i][2] = -predictor[i]*exp(-Lambda[i]*predictor[i])/(exp(-Lambda[i]*predictor[i])-exp(-Lambda[second[i]]*predictor[i]));
                else
                {
                    if (second[i]<=nn)
                        aa[i][2] = -predictor[i];
                    else
                        aa[i][4] = -predictor[i];
                }
            }
            else
            {
                if (second[i]>=first)
                    aa[i][3] = predictor[i]*exp(-Lambda[i]*predictor[i])/(exp(-Lambda[second[i]]*predictor[i])-exp(-Lambda[i]*predictor[i]));
                else
                    aa[i][3] = predictor[i]*exp(-Lambda[i]*predictor[i])/(1-exp(-Lambda[i]*predictor[i]));
            }
        }
    }
    
    
    for (i=first;i<=last;i++)
    {
        if (second[i]==0)
            bb[i][1] = SQR(predictor[i])*exp(-Lambda[i]*predictor[i])/SQR(1-exp(-Lambda[i]*predictor[i]));
        else
        {
            if (second[i]>i)
            {
                if (second[i]<=last)
                    bb[i][2] = SQR(predictor[i])*exp(-(Lambda[second[i]]-Lambda[i])*predictor[i])/SQR(1-exp(-(Lambda[second[i]]-Lambda[i])*predictor[i]));
                else
                    bb[i][2] = 0;
            }
            else
            {
                if (second[i]>=first)
                    bb[i][3] = SQR(predictor[i])*exp(-(Lambda[i]-Lambda[second[i]])*predictor[i])/SQR(1-exp(-(Lambda[i]-Lambda[second[i]])*predictor[i]));
                else
                    bb[i][3] = SQR(predictor[i])*exp(-Lambda[i]*predictor[i])/SQR(1-exp(-Lambda[i]*predictor[i]));
            }
        }
        
        
    }
    
    for (i=1;i<=nn;i++)
    {
        for (j=1;j<=m1+1;j++)
            aa[i][j] /=nn;
        
        for (j=1;j<=m1;j++)
            bb[i][j] /=nn;
    }
}


void swap(double *x,double *y)
{
    double temp;
    temp=*x;
    *x=*y;
    *y=temp;
}


void rank2(int n, int indx[], int irank[])
{
    int j;
    
    for (j=1;j<=n;j++) irank[indx[j]]=j;
}


void indexx(int n, double arr[], int indx[])
{
    int i,indxt,ir=n,itemp,j,k,l=1;
    int jstack=0,*istack;
    double a;
    
    istack=new int[NSTACK+1];
    
    for (j=1;j<=n;j++) indx[j]=j;
    for (;;)
    {
        if (ir-l < M)
        {
            for (j=l+1;j<=ir;j++)
            {
                indxt=indx[j];
                
                a=arr[indxt];
                for (i=j-1;i>=1;i--)
                {
                    if (arr[indx[i]] <= a) break;
                    indx[i+1]=indx[i];
                }
                
                indx[i+1]=indxt;
                
            }
            if (jstack == 0) break;
            ir=istack[jstack--];
            l=istack[jstack--];
        }
        else
        {
            k=(l+ir) >> 1;
            SWAP(indx[k],indx[l+1]);
            if (arr[indx[l+1]] > arr[indx[ir]])
            {
                SWAP(indx[l+1],indx[ir])
            }
            if (arr[indx[l]] > arr[indx[ir]])
            {
                SWAP(indx[l],indx[ir])
            }
            if (arr[indx[l+1]] > arr[indx[l]]) {
                SWAP(indx[l+1],indx[l])
            }
            i=l+1;
            j=ir;
            indxt=indx[l];
            a=arr[indxt];
            for (;;) {
                do i++; while (arr[indx[i]] < a);
                do j--; while (arr[indx[j]] > a);
                if (j < i) break;
                SWAP(indx[i],indx[j])
            }
            indx[l]=indx[j];
            indx[j]=indxt;
            jstack += 2;
            if (jstack > NSTACK) printf("NSTACK too small in indexx.");
            if (ir-i+1 >= j-l) {
                istack[jstack]=ir;
                istack[jstack-1]=i;
                ir=j-1;
            }
            else
            {
                istack[jstack]=j-1;
                istack[jstack-1]=l;
                l=i;
            }
        }
    }
    delete[] istack;
}



void sortcens(int n, int *nn, int m1, double vv[])
{
    int i,j,k;
    int    *index,*irank,*second2;
    double *vv2;
    
    
    index = new int[2*n+2];
    irank = new int[2*n+2];
    second2 = new int[2*n+2];
    vv2 = new double[2*n+2];
    
    j = 0;
    for (i=1;i<=n;i++)
    {
        if (ind[i]==1)
        {
            j++;
            vv[j] = tt[i];
            for (k=1;k<=m1;k++)
                zz2[j][k]=zz[i][k];
            second[j] = 0;
        }
        else
        {
            if (ind[i]==2)
            {
                j++;
                vv[j] = tt[i];
                for (k=1;k<=m1;k++)
                    zz2[j][k]=zz[i][k];
                second[j] = j+1;
                j++;
                vv[j] = uu[i];
                for (k=1;k<=m1;k++)
                    zz2[j][k]=zz[i][k];
                second[j] =j-1;
            }
            else
            {
                j++;
                vv[j] = uu[i];
                for (k=1;k<=m1;k++)
                    zz2[j][k]=zz[i][k];
                second[j] = 2*n+1;
            }
        }
    }
    
    *nn=j;
    
    indexx(*nn,vv,index);
    rank2(*nn,index,irank);
    
    for (i=1;i<=*nn;i++)
        if (second[i]==2*n+1)
            second[i]=*nn+1;
    
    irank[0]=0;
    irank[*nn+1]=*nn+1;
    
    for (i=1;i<=*nn;i++)
    {
        second2[irank[i]]=irank[second[i]];
        vv2[irank[i]]=vv[i];
        for (k=1;k<=m1;k++)
            zz[irank[i]][k]=zz2[i][k];
    }
    
    for (i=1;i<=*nn;i++)
    {
        second[i]=second2[i];
        vv[i]=vv2[i];
    }
    
    delete[] index;  delete[] irank; delete[] second2; delete[] vv2;
    
}


int compare(const void *a, const void *b)
{
    double x = *(double*)a;
    double y = *(double*)b;
    
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}


//
//  Hooke_Jeeves.cpp
//  From TOMS178.cpp

//****************************************************************************80
//
//  Purpose:
//
//    BEST_NEARBY looks for a better nearby point, one coordinate at a time.
//
//  Modified:
//
//    12 February 2008
//
//  Author:
//
//    The ALGOL original is by Arthur Kaupe.
//    C version by Mark Johnson
//    C++ version by John Burkardt
//
//  Reference:
//
//    M Bell, Malcolm Pike,
//    Remark on Algorithm 178: Direct Search,
//    Communications of the ACM,
//    Volume 9, Number 9, September 1966, page 684.
//
//    Robert Hooke, Terry Jeeves,
//    Direct Search Solution of Numerical and Statistical Problems,
//    Journal of the ACM,
//    Volume 8, Number 2, April 1961, pages 212-229.
//
//    Arthur Kaupe,
//    Algorithm 178:
//    Direct Search,
//    Communications of the ACM,
//    Volume 6, Number 6, June 1963, page 313.
//
//    FK Tomlin, LB Smith,
//    Remark on Algorithm 178: Direct Search,
//    Communications of the ACM,
//    Volume 12, Number 11, November 1969, page 637-638.
//
//  Parameters:
//
//    Input, double DELTA(NVARS), the size of a step in each direction.
//
//    Input/output, double POINT(NVARS); on input, the current candidate.
//    On output, the value of POINT may have been updated.
//
//    Input, double PREVBEST, the minimum value of the function seen
//    so far.
//
//    Input, int NVARS, the number of variables.
//
//    Input, F, the name of the function routine,
//    which should have the form:
//      double f ( double x[], int n )
//
//    Input/output, int *FUNEVALS, the number of function evaluations.
//
//    Output, double BEST_NEARBY, the minimum value of the function seen
//    after checking the nearby neighbors.
//


double best_nearby (int m, double delta[], double point[], double prevbest,
                    double f(int m, double alpha[]), int *funevals)
{
    double ftmp,minf,*z;
    int i;
    
    z = new double[m];
    
    minf = prevbest;
    
    for ( i = 0; i < m; i++ )
        z[i] = point[i];
    
    for ( i = 0; i < m; i++ )
    {
        z[i] = point[i] + delta[i];
        
        ftmp = f(m,z);
        *funevals +=1;
        
        if (ftmp < minf)
            minf = ftmp;
        else
        {
            delta[i] = - delta[i];
            z[i] = point[i] + delta[i];
            ftmp = f(m,z);
            *funevals +=1;
            
            if ( ftmp < minf )
                minf = ftmp;
            else
                z[i] = point[i];
        }
    }
    
    for ( i = 0; i < m; i++ )
        point[i] = z[i];
    
    delete [] z;
    
    return minf;
}


int hooke(int m, double startpt[], double endpt[], double rho, double eps,
          int itermax, double f(int m, double alpha[]))
{
    double *delta,fbefore;
    int i,iter,keep,funevals,count;
    double newf,*newx,steplength,tmp;
    double *xbefore;
    
    delta = new double[m];
    newx = new double[m];
    xbefore = new double[m];
    
    for ( i = 0; i < m; i++ )
        xbefore[i] = newx[i] = startpt[i];
    
    for ( i = 0; i < m; i++ )
        delta[i] = rho;
    
    
    //printf("\n %16.10f %15.10f %15.10f\n \n", newx[0],newx[1],newx[2]);
    
    funevals = 0;
    steplength = rho;
    iter = 0;
    
    
    fbefore = f(m,newx);
    funevals = funevals + 1;
    newf = fbefore;
    
    while (iter<itermax && eps<steplength)
    {
        iter++;
        
        for ( i = 0; i < m; i++ )
            newx[i] = xbefore[i];
        
        //  Find best new alpha.
        
        newf = best_nearby(m,delta,newx,fbefore,f,&funevals);
        //
        //  If we made some improvements, pursue that direction.
        //
        keep = 1;
        count=0;
        
        //while (newf<fbefore - 0.1*SQR(steplength) && keep == 1 && count<=100)
        while (newf<fbefore && keep == 1 && count<=100)
        {
            count++;
            for ( i = 0; i < m; i++ )
            {
                //
                //  Arrange the sign of DELTA.
                //
                if ( newx[i] <= xbefore[i] )
                    delta[i] = - fabs(delta[i]);
                else
                    delta[i] = fabs(delta[i]);
                //
                //  Now, move further in this direction.
                //
                tmp = xbefore[i];
                xbefore[i] = newx[i];
                newx[i] = newx[i] + newx[i] - tmp;
            }
            
            fbefore = newf;
            
            newf = best_nearby(m,delta,newx,fbefore,f,&funevals);
            //
            //  If the further (optimistic) move was bad...
            //
            //if (newf >= fbefore - 0.1*SQR(steplength))
            if (newf >= fbefore)
                break;
            //
            //  Make sure that the differences between the new and the old points
            //  are due to actual displacements; beware of roundoff errors that
            //  might cause NEWF < FBEFORE.
            //
            /*keep = 0;
             
             for ( i = 0; i < m; i++ )
             {
             if ( 0.5 * fabs(delta[i]) < fabs(newx[i]-xbefore[i]))
             {
             keep = 1;
             break;
             }
             }*/
        }
        
        //if (eps <= steplength && newf >= fbefore - 0.1*SQR(steplength))
        if (eps <= steplength && newf >= fbefore)
        {
            steplength = steplength * rho;
            for ( i = 0; i < m; i++ )
                delta[i] = delta[i] * rho;
        }
        
    }
    
    for ( i = 0; i < m; i++ )
        endpt[i] = xbefore[i];
    
    delete [] delta;
    delete [] newx;
    delete [] xbefore;
    
    return funevals;
}




