#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <functional>
#include <iostream>
#include <vector>
using namespace std;

// overlap
bool contain(double Asx, double Asy, double Asex, double Asey, double cx,
             double cy) {
  // cx = Asx + Asex*tx
  // cy = Asy + Asey*ty
  double tx = (cx - Asx) / Asex;
  double ty = (cy - Asy) / Asey;
  return tx == ty && 0 <= tx && tx <= 1;
}

// sequential code
void intersectVector(double Asx,           // lineSegment1 start x
                     double Asy,           // lineSegment1 start y
                     double Aex,           // lineSegment1 end x
                     double Aey,           // lineSegment1 end y
                     double* Barrsx,       // lineSegment2 start x
                     double* Barrsy,       // lineSegment2 start y
                     double* Barrex,       // lineSegment2 end x
                     double* Barrey,       // lineSegment2 end y
                     double* Rarrx,        // point of intersection x
                     double* Rarry,        // point of intersection y
                     bool* Rintersection,  // judge the intersection results
                     bool* Roverlap,       // judge overlap
                     int size              // size of array
) {
  // vector A
  double Asex = Aex - Asx;
  double Asey = Aey - Asy;

  // array line segmentB
  for (int i = 0; i < size; i++) {
    double Bsx = Barrsx[i];
    double Bsy = Barrsy[i];
    double Bex = Barrex[i];
    double Bey = Barrey[i];
    double Bsex = Bex - Bsx;
    double Bsey = Bey - Bsy;

    // denominator
    double denom = Asey * Bsex - Asex * Bsey;

    // numerator for s
    double s1 = Asx * Bsey + Bsx * (Asy - Bey) + Bex * (Bsy - Asy);
    double s = s1 / denom;

    // numerator for t
    double t1 = Asx * (Aey - Bsy) + Aex * (Bsy - Asy) - Bsx * Asey;
    double t = t1 / denom;

    // point of intersection
    double Rx = Asx + s * Asex;
    double Ry = Asy + s * Asey;

    // Segments can't be parallel
    Rintersection[i] = (denom != 0 & s >= 0 & s <= 1 & t >= 0 & t <= 1);
    Rarrx[i] = Rx;
    Rarry[i] = Ry;

    // overlap
    // sx,sy on the line or ex,ey on the line
    Roverlap[i] = (denom == 0 & (contain(Asx, Asy, Aex, Aey, Bsx, Bsy) |
                                 contain(Asx, Asy, Aex, Aey, Bex, Bey) |
                                 contain(Bsx, Bsy, Bex, Bey, Asx, Asy) |
                                 contain(Bsx, Bsy, Bex, Bey, Aex, Aey)));
  }
}

const __m256d zero = _mm256_set1_pd(0);
const __m256d one = _mm256_set1_pd(1);

__m256d containSIMD(__m256d vAsx, __m256d vAsy, __m256d vAsex, __m256d vAsey,
                    __m256d vcx, __m256d vcy) {
  __m256d tx = _mm256_div_pd(_mm256_sub_pd(vcx, vAsx), vAsex);
  __m256d ty = _mm256_div_pd(_mm256_sub_pd(vcy, vAsy), vAsey);
  // tx == ty
  __m256d c1 = _mm256_cmp_pd(tx, ty, _CMP_EQ_OS);
  // s>=0
  __m256d c2 = _mm256_cmp_pd(tx, zero, _CMP_GE_OS);
  // s<=1
  __m256d c3 = _mm256_cmp_pd(tx, one, _CMP_LE_OS);

  return _mm256_and_pd(c1, _mm256_and_pd(c2, c3));
}

// simd code
void intersectSIMD(double Asx,           // lineSegmentA start x
                   double Asy,           // lineSegmentA start y
                   double Aex,           // lineSegmentA end x
                   double Aey,           // lineSegmentA end y
                   double* Barrsx,       // lineSegmentB start x
                   double* Barrsy,       // lineSegmentB start y
                   double* Barrex,       // lineSegmentB end x
                   double* Barrey,       // lineSegmentB end y
                   double* Rarrx,        // point of intersection x
                   double* Rarry,        // point of intersection y
                   bool* Rintersection,  // judge the intersection results
                   bool* Roverlap,       // judge overlap
                   int size              // size of array
) {
  // line segmentA
  __m256d vAsx = _mm256_set1_pd(Asx);
  __m256d vAsy = _mm256_set1_pd(Asy);
  __m256d vAex = _mm256_set1_pd(Aex);
  __m256d vAey = _mm256_set1_pd(Aey);

  // vector A
  __m256d vAsex = _mm256_sub_pd(vAex, vAsx);
  __m256d vAsey = _mm256_sub_pd(vAey, vAsy);

  for (int i = 0; i < size; i = i + 4) {
    // line segment B
    __m256d vBsx = _mm256_load_pd(Barrsx + i);
    __m256d vBsy = _mm256_load_pd(Barrsy + i);
    __m256d vBex = _mm256_load_pd(Barrex + i);
    __m256d vBey = _mm256_load_pd(Barrey + i);

    // vector B
    __m256d vBsex = _mm256_sub_pd(vBex, vBsx);
    __m256d vBsey = _mm256_sub_pd(vBey, vBsy);

    // denominator
    // D = vAsey*vBsex - vAsex*vBsey
    __m256d denom =
        _mm256_sub_pd(_mm256_mul_pd(vAsey, vBsex), _mm256_mul_pd(vAsex, vBsey));

    // numerator for s
    __m256d s1 = _mm256_add_pd(_mm256_mul_pd(vAsx, vBsey),
                               _mm256_mul_pd(vBsx, _mm256_sub_pd(vAsy, vBey)));

    __m256d s2 = _mm256_mul_pd(vBex, _mm256_sub_pd(vBsy, vAsy));
    __m256d s3 = _mm256_add_pd(s1, s2);
    __m256d s = _mm256_div_pd(s3, denom);

    // numerator for t
    __m256d t1 = _mm256_add_pd(_mm256_mul_pd(vAsx, _mm256_sub_pd(vAey, vBsy)),
                               _mm256_mul_pd(vAex, _mm256_sub_pd(vBsy, vAsy)));
    __m256d t2 = _mm256_mul_pd(vBsx, vAsey);
    __m256d t3 = _mm256_sub_pd(t1, t2);
    __m256d t = _mm256_div_pd(t3, denom);

    // point of intersection
    __m256d vRx =
        _mm256_add_pd(vAsx, _mm256_mul_pd(s, _mm256_sub_pd(vAex, vAsx)));

    __m256d vRy =
        _mm256_add_pd(vAsy, _mm256_mul_pd(s, _mm256_sub_pd(vAey, vAsy)));

    // compare

    // denom!=0
    __m256d c1 = _mm256_cmp_pd(denom, zero, _CMP_NEQ_OS);
    // s>=0
    __m256d c2 = _mm256_cmp_pd(s, zero, _CMP_GE_OS);
    // s<=1
    __m256d c3 = _mm256_cmp_pd(s, one, _CMP_LE_OS);
    // t>=0
    __m256d c4 = _mm256_cmp_pd(t, zero, _CMP_GE_OS);
    // t<=1
    __m256d c5 = _mm256_cmp_pd(t, one, _CMP_LE_OS);

    __m256d intersect = _mm256_and_pd(
        c1, _mm256_and_pd(c2, _mm256_and_pd(c3, _mm256_and_pd(c4, c5))));

    // overlap
    __m256d overlap = _mm256_and_pd(
        _mm256_cmp_pd(denom, zero, _CMP_EQ_OS),  // D = 0
        _mm256_or_pd(
            _mm256_or_pd(containSIMD(vAsx, vAsy, vAsex, vAsey, vBsx, vBsy),
                         containSIMD(vAsx, vAsy, vAsex, vAsey, vBex, vBey)),
            _mm256_or_pd(containSIMD(vBsx, vBsy, vBsex, vBsey, vAsx, vAsy),
                         containSIMD(vBsx, vBsy, vBsex, vBsey, vAex, vAey))));
    // make sure every data meets the conditions
    for (int j = 0; j < 4; j++) {
      Rintersection[i + j] = (intersect[j] != 0.0);
      Roverlap[i + j] = (overlap[j] != 0.0);
    }

    _mm256_store_pd(Rarrx + i, vRx);
    _mm256_store_pd(Rarry + i, vRy);
  }
}

// function to do performance test
void performanceTest() {
  const int size = 8 * 99999;
  const int count = 1000;

  // linesegment1
  double sx = 0, sy = 1, ex = 10, ey = 2;

  // memory allocation for arrays vsx,vsy,vex,vey using _mm_malloc
  double* vsx = (double*)_mm_malloc(size * sizeof(double), 32);
  double* vsy = (double*)_mm_malloc(size * sizeof(double), 32);
  double* vex = (double*)_mm_malloc(size * sizeof(double), 32);
  double* vey = (double*)_mm_malloc(size * sizeof(double), 32);

  // linesegment2
  for (int i = 0; i < size / 2; i++) {
    vsx[i] = -1;
    vsy[i] = 0;
    vex[i] = 10 * ((double)i / (double)size);
    vey[i] = 3;
  }

  // memory allocation for arrays vpix,vpiy using _mm_malloc
  double* vpix = (double*)_mm_malloc(size * sizeof(double), 32);
  double* vpiy = (double*)_mm_malloc(size * sizeof(double), 32);

  bool intersectSuccess[size];
  bool overlap[size];
  // get the running time
  clock_t time_start;
  double time_spent, time_spent_simd;

  // time of intersectVector

  time_start = clock();
  for (int i = 0; i < count; i++) {
    intersectVector(sx, sy, ex, ey, vsx, vsy, vsx, vsy, vpix, vpiy,
                    intersectSuccess, overlap, size);
  }
  time_spent = (double)(clock() - time_start) / CLOCKS_PER_SEC;

  printf("general time: %f s\n", time_spent);

  // time of intersectSIMD

  time_start = clock();
  for (int i = 0; i < count; i++) {
    intersectSIMD(sx, sy, ex, ey, vsx, vsy, vsx, vsy, vpix, vpiy,
                  intersectSuccess, overlap, size);
  }

  time_spent_simd = (double)(clock() - time_start) / CLOCKS_PER_SEC;

  printf("SIMD time: %f s\n", time_spent_simd);

  printf("speed increased %.2f%\n",
         (time_spent - time_spent_simd) / time_spent * 100);

  // de-allocating memory for vsx, vsy, vex, vey, vpix, vpiy using _mm_free
  _mm_free(vsx);
  _mm_free(vsy);
  _mm_free(vex);
  _mm_free(vey);
  _mm_free(vpix);
  _mm_free(vpiy);
}

// function to do functional test
void test() {
  const int size = 7;
  // linesegment1
  double Asx = 1;
  double Asy = 1;
  double Aex = 3;
  double Aey = 3;

  // linesegment2
  // linesegment(1,3,3,1)and(1,1,3,3)should intersect at (2,2)
  // linesegment(1,3,3,2)and(1,1,3,3)should intersect at (2.33333,2.33333)
  // linesegment(1,2,2,1)and(1,1,3,3)should intersect at (1.5,1.5)
  // linesegment(1,2,3,10)and(1,1,3,3)should be no intersection
  // inesegment(1,2,5,8)and(1,1,3,3)should be no intersection
  // inesegment(1,1,3,3)and(1,1,3,3)should be overlapped
  // inesegment(0,0,2,2)and(1,1,3,3)should be overlapped
  double varrsxs[size] = {1, 1, 1, 1, 1, 1, 0};
  double varrsys[size] = {3, 3, 2, 2, 2, 1, 0};
  double varrexs[size] = {3, 3, 2, 3, 5, 3, 2};
  double varreys[size] = {1, 2, 1, 10, 8, 3, 2};

  // memory allocation for arrays arrsxs,arrsys,arrexs,arreys using
  // _mm_malloc
  double* Barrsx = (double*)_mm_malloc(size * sizeof(double), 32);
  double* Barrsy = (double*)_mm_malloc(size * sizeof(double), 32);
  double* Barrex = (double*)_mm_malloc(size * sizeof(double), 32);
  double* Barrey = (double*)_mm_malloc(size * sizeof(double), 32);

  for (int i = 0; i < size; i++) {
    Barrsx[i] = varrsxs[i];
    Barrsy[i] = varrsys[i];
    Barrex[i] = varrexs[i];
    Barrey[i] = varreys[i];
  }

  // memory allocation for arrays pixs,piys using _mm_malloc
  double* Rx = (double*)_mm_malloc(size * sizeof(double), 32);
  double* Ry = (double*)_mm_malloc(size * sizeof(double), 32);

  bool Rintersection[size];
  bool Roverlap[size];
  // test intersectVector
  intersectVector(Asx, Asy, Aex, Aey, Barrsx, Barrsy, Barrex, Barrey, Rx, Ry,
                  Rintersection, Roverlap, size);

  for (int i = 0; i < size; i++) {
    if (Rintersection[i]) {
      printf("general intersect at: (%.2f,%.2f) \n", Rx[i], Ry[i]);
    } else {
      printf("no intersection");
      if (Roverlap[i]) {
        printf(" but overlap");
      }
      cout << endl;
    }
  }
  cout << endl;

  // test intersectSIMD
  intersectSIMD(Asx, Asy, Aex, Aey, Barrsx, Barrsy, Barrex, Barrey, Rx, Ry,
                Rintersection, Roverlap, size);
  for (int i = 0; i < size; i++) {
    if (Rintersection[i]) {
      printf("SIMD intersect at: (%.2f,%.2f) \n", Rx[i], Ry[i]);
    } else {
      printf("no intersection");
      if (Roverlap[i]) {
        printf(" but overlap");
      }
      cout << endl;
    }
  }

  // de-allocating memory for arrsxs,arrxys,arrexs,arrxys,pixs,piys using
  // _mm_free
  _mm_free(Barrsx);
  _mm_free(Barrsy);
  _mm_free(Barrex);
  _mm_free(Barrey);
  _mm_free(Rx);
  _mm_free(Ry);
}

// read vertex
void read_vertices(const char* filename, int& size, double*& x_list,
                   double*& y_list) {
  FILE* fp = fopen(filename, "r");
  fscanf(fp, "%d", &size);

  // memory allocation for arrays x_list, y_list using _mm_malloc
  x_list = (double*)_mm_malloc(size * sizeof(double), 32);
  y_list = (double*)_mm_malloc(size * sizeof(double), 32);
  for (int i = 0; i < size; i++) {
    fscanf(fp, "%lf,%lf", x_list + i, y_list + i);
  }
  printf("%s done reading\n", filename);
  printf("read %d vertices\n", size);
  for (int i = 0; i < 5; i++) {
    printf("(%.1f, %.1f)\n", x_list[i], y_list[i]);
  }
  printf("...\n");
  fclose(fp);
}

// write vertex
void write_vertices(const char* filename, vector<pair<double, double>>& vec) {
  FILE* fp = fopen(filename, "w");
  fprintf(fp, "%d\n", (int)vec.size());
  // print intersections
  for (int i = 0; i < vec.size(); i++) {
    fprintf(fp, "%f,%f\n", vec[i].first, vec[i].second);
  }
  fclose(fp);
}

// define a function type
using IntersectMethod = void (*)(
    double Asx,           // lineSegment1 start x
    double Asy,           // lineSegment1 start y
    double Aex,           // lineSegment1 end x
    double Aey,           // lineSegment1 end y
    double* Barrsx,       // lineSegment2 start x
    double* Barrsy,       // lineSegment2 start y
    double* Barrex,       // lineSegment2 end x
    double* Barrey,       // lineSegment2 end y
    double* Rarrx,        // point of intersection x
    double* Rarry,        // point of intersection y
    bool* Rintersection,  // judge the intersection results
    bool* Roverlap,       // judge overlap
    int size              // size of array
);

double polyTest(const char* inputMore,  // file with more points
                const char* inputLess,  // file with less points
                const char* output,     // output results
                IntersectMethod method) {
  clock_t time_start = clock();
  // s,c data size›
  int s_size, c_size;
  // s,c data array
  double *s_x_list, *s_y_list, *c_x_list, *c_y_list;
  read_vertices(inputMore, s_size, s_x_list, s_y_list);
  read_vertices(inputLess, c_size, c_x_list, c_y_list);

  // end points of x,y
  double* se_x_list = (double*)_mm_malloc((s_size - 1) * sizeof(double), 32);
  double* se_y_list = (double*)_mm_malloc((s_size - 1) * sizeof(double), 32);
  for (int i = 0; i < s_size - 1; i++) {
    se_x_list[i] = s_x_list[i + 1];
    se_y_list[i] = s_y_list[i + 1];
  }

  // store intersections
  vector<pair<double, double>> intersections;

  double* vpix = (double*)_mm_malloc((s_size - 1) * sizeof(double),
                                     32);  // x coordinates of intersections
  double* vpiy = (double*)_mm_malloc((s_size - 1) * sizeof(double),
                                     32);  // y coordinates of intersections

  bool* intersectSuccess = new bool[s_size - 1];  // store intersection results
  bool* overlap = new bool[s_size - 1];           // store intersection results

  for (int i = 0; i < c_size - 1; i++) {
    if (i % 10000 == 0) printf("polyTest progress %d/%d\n", i + 1, c_size - 1);
    method(c_x_list[i], c_y_list[i], c_x_list[i + 1],
           c_y_list[i + 1],                           // lineSegment1
           s_x_list, s_y_list, se_x_list, se_y_list,  // lineSegment2
           vpix, vpiy, intersectSuccess, overlap,     // intersections
           s_size - 1                                 // numbers of lineSegments
    );
    // traverse intersections
    for (int j = 0; j < s_size - 1; j++) {
      if (intersectSuccess[j]) {
        intersections.push_back(make_pair(vpix[j], vpiy[j]));
      }
    }
  }

  // de-allocating memory for vpix, vpiy, intersectSuccess using _mm_free
  _mm_free(vpix);
  _mm_free(vpiy);
  free(intersectSuccess);

  write_vertices(output, intersections);
  double time_spent = (double)(clock() - time_start) / CLOCKS_PER_SEC;
  printf("Spent time: %f s\n", time_spent);
  return time_spent;
}

int main() {
  cout << "------ run test ------" << endl;
  test();

  cout << endl;
  cout << "------ run performanceTest ------" << endl;
  performanceTest();

  const char* inputMore = "poly/s.txt";
  const char* inputLess = "poly/c.txt";
  cout << endl;
  cout << "------ run polyTest by sequential------" << endl;
  double sequentialTime = polyTest(
      inputMore, inputLess, "poly/sequential_output.txt", intersectVector);

  cout << endl;
  cout << "------ run polyTest by SIMD------" << endl;
  double simdTime =
      polyTest(inputMore, inputLess, "poly/simd_output.txt", intersectSIMD);

  printf("speed increased: %.2f %\n",
         ((sequentialTime - simdTime) / sequentialTime) * 100);
  return 0;
}