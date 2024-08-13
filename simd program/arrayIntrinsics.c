#include<stdio.h>
#include<immintrin.h>
#include<stdlib.h>
#include<time.h>
#include<assert.h>


void display(int n, double *arr)
{
    //printf("%f %f %f %f %f %f \n", arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]);
    int i;
    for( i = 0; i < 16; i++)
    {
       printf("%.2f ", arr[i]);
    }
    printf("\n");
}

// Problem 2.1
void vecAdd(int n, double *A, double *B, double *C)
{
  int i;
  for(i = 0; i<n; i++)
  {
    C[i] = C[i] + A[i] * B[i];
  }
}

// Problem 2.2
double vecAdd10(int n, double *A, double *B)
{
  int i; 
  double c = 0;
  for(i = 0; i<n; i++)
  {
    c = c + A[i] * B[i];
  }
  return c;
}

// Problem 2.3
void vecAddV2(int n, double *A, double *B, double *C)
{
  int i,j;
  double a0,c0;

  for(i = 0; i<n; i++)
  {
    c0 = C[i];
    a0 = A[i];
    
    for(j = 0; j<n; j++)
    {
       c0 = c0 + a0 * B[j];
    }
    C[i] = c0;
  }
}

// Answer 2.1
void vecAddIntrin(int n, double *A, double *B, double *C)
{
  int i;
  for(i = 0; i<n; i = i+4)
  {
     __m256d c0 = _mm256_load_pd(C+i); 

     c0 = _mm256_add_pd( c0, _mm256_mul_pd( _mm256_load_pd(A+i), 
                                                _mm256_load_pd(B+i)) );
                               
     _mm256_store_pd( C+i , c0);
  }
}

// Duplicate answer to problem 2.1. I discussed this solution to problem 2.1 in class
// This answer is easy to read; the algorithm is same as the previous answer
void vecAddIntrinClass(int n, double *A, double *B, double *C)
{
  int i;
  for(i = 0; i<n; i = i+4)
  {
     __m256d c0 = _mm256_load_pd(C+i);    
     __m256d a0 = _mm256_load_pd(A+i);
     __m256d b0 = _mm256_load_pd(B+i);
     
     __m256d multipliedA0B0 =  _mm256_mul_pd( a0, b0);
      
     c0 = _mm256_add_pd( c0, multipliedA0B0);
                               
     _mm256_store_pd( C+i , c0);
  }
}


// Answer 2.2
double vecAdd10Intrin(int n, double *A, double *B)
{
    int i;
	double *c = (double *)_mm_malloc( 4 * sizeof(double), 32);
	c[0] = 0.0;
	c[1] = 0.0;
	c[2] = 0.0;
	c[3] = 0.0;

	__m256d c0 = _mm256_load_pd(c);
	for(i = 0; i< n ; i = i+4)
	{
  		c0 = _mm256_add_pd( c0, _mm256_mul_pd( _mm256_load_pd(A+i), 
                                                _mm256_load_pd(B+i)) );   
	}
 	
 	_mm256_store_pd( c , c0);
 	
 	return c[0] + c[1] + c[2] + c[3];   
}

/*
A:   1   1   1   1    
B:   2   2   2   2
C:   8   8   8   8
*/

// Answer to problem 2.3
void vecAddV2IntrinBC(int n, double *A, double *B, double *C)
{
  int i,j;

  __m256d a0, c0;
  
  for(i = 0; i<n; i = i+4)
  {
      c0 = _mm256_load_pd(C+i); 
      a0 = _mm256_load_pd(A+i); 
      
      for(j = 0; j<n; j = j+1)
      {
        c0 = _mm256_add_pd( c0, _mm256_mul_pd( a0, _mm256_broadcast_sd(B+j)) );
      }     
     
      _mm256_store_pd( C+i , c0);
  }
}

void set(double *arr, int length, double val)
{
    int i;
    for( i = 0; i<length; i++)
    {
       arr[i] = val;
    }
}

void initialize(double *arr, int length)
{
    int i;
    for( i = 0; i<length; i++)
    {
       arr[i] = i+1;
    }
}

int main()
{
   //int NUM_ELEMENTS = 32;
   //int NUM_ELEMENTS = 1024*24*2;
   int NUM_ELEMENTS = 1024*1024*128;
   //int NUM_ELEMENTS = 1024*1024*128;
   
   printf("NUM_ELEMENTS %d, sizeof(double)=%lu \n ", NUM_ELEMENTS, sizeof(double));
   
   size_t N_pd = (NUM_ELEMENTS*8)/sizeof(double);

   double *data_A = (double*)_mm_malloc(N_pd*sizeof(double), 32);
   double *data_B = (double*)_mm_malloc(N_pd*sizeof(double), 32);
   double *data_C = (double*)_mm_malloc(N_pd*sizeof(double), 32);

   if(data_A == NULL || data_B == NULL || data_C == NULL)
   {
     printf("Error \n");
     return 1;
   }
   
   set(data_A, NUM_ELEMENTS, 2.0);
   //initialize(data_A, NUM_ELEMENTS);
   set(data_B, NUM_ELEMENTS, 3.0);
   //initialize(data_B, NUM_ELEMENTS);
   set(data_C, NUM_ELEMENTS, 1.0);
   
   clock_t begin = clock();
   
   //vecAdd(NUM_ELEMENTS, data_A, data_B, data_C);
   
   double normalDotProduct = vecAdd10(NUM_ELEMENTS, data_A, data_B);
   printf(" vectorDotProduct %f \n", normalDotProduct);
   
   //vecAddV2(NUM_ELEMENTS, data_A, data_B, data_C);
   
   clock_t end = clock();
   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

   //display(NUM_ELEMENTS, data_C);

   // reset the values in array data_C
   set(data_C, NUM_ELEMENTS, 1.0);     
   printf("Non-intrinsic Time %f \n", time_spent);
   
   begin = clock();
   
   //vecAddIntrin(NUM_ELEMENTS, data_A, data_B, data_C);
   //vecAddIntrinClass(NUM_ELEMENTS, data_A, data_B, data_C);
   
   //vecAddV2Intrin(NUM_ELEMENTS, data_A, data_B, data_C);
   
   double vectorDotProduct = vecAdd10Intrin(NUM_ELEMENTS, data_A, data_B);
   printf(" vectorDotProduct %f \n", vectorDotProduct);
   
   //vecAddV2IntrinBC( NUM_ELEMENTS, data_A, data_B, data_C);
   
   end = clock();
   
   time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
      
   //display(NUM_ELEMENTS, data_C);
   
   printf("Instrinsics Vectorization Time %f \n", time_spent);
   
   _mm_free(data_A);
   _mm_free(data_B);
   _mm_free(data_C);
   
   return 0;
} 