#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    return mat->data[row * (mat->cols) +col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    mat->data[row * (mat -> cols) + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
        if (rows <= 0 || cols <= 0)
                return -1;
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
        matrix *alloced_struct_ptr = (matrix *) malloc(sizeof(matrix));
        //this is, of course, the alloc-failure part
        if (alloced_struct_ptr == NULL)//this may need to be *mat/**mat? 
                return -2;
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
        double *alloced_data_ptr = (double *) calloc(rows*cols, sizeof(double));
        if (alloced_data_ptr == NULL)
                return -2;
        alloced_struct_ptr->data = alloced_data_ptr;
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
        alloced_struct_ptr->rows = rows;
        alloced_struct_ptr->cols = cols;
    // 5. Set the parent field to NULL, since this matrix was not created from a slice.
        alloced_struct_ptr->parent = NULL;
    // 6. Set the ref_cnt field to 1.
        alloced_struct_ptr->ref_cnt = 1;
    // 7. Store the address of the allocated matrix struct at the location mat is pointing at.
        *mat = alloced_struct_ptr;//this line of code may be problematic?
    // 8. Return 0 upon success.
        return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer mat is NULL, return.
    // 2. If mat has no parent: decrement its ref_cnt field by 1. If the ref_cnt field becomes 0, then free mat and its data field.
    // 3. Otherwise, recursively call deallocate_matrix on mat's parent, then free mat.
    if (mat == NULL) {
        return;
    }
    if(mat->parent == NULL){
        mat->ref_cnt -= 1;
        if(mat->ref_cnt==0) {
            free(mat->data);  
            free(mat);
        } 
    }else {
        deallocate_matrix(mat->parent);  
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
        if (rows <= 0 || cols <= 0)
                return -1;
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
        matrix *alloced_struct_4_ptr = (matrix *) malloc(sizeof(matrix));
        if (alloced_struct_4_ptr == NULL)
                return -2;
    // 3. Set the data field of the new struct to be the data field of the from struct plus offset.
        alloced_struct_4_ptr->data = from->data + offset;
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
        alloced_struct_4_ptr->rows = rows;
        alloced_struct_4_ptr->cols = cols;
    // 5. Set the parent field of the new struct to the from struct pointer.
        alloced_struct_4_ptr->parent = from;
    // 6. Increment the ref_cnt field of the from struct by 1.
        from->ref_cnt += 1;
    // 7. Store the address of the allocated matrix struct at the location mat is pointing at.
        *mat = alloced_struct_4_ptr;
    // 8. Return 0 upon success.
        return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    int rows = mat->rows;
    int cols = mat->cols;
    int boundary = cols/4*4;
    double * data = mat->data;
    __m256d vals = _mm256_set1_pd(val);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++){
        for (int j = 0; j<boundary; j+= 4) {
                int offset = i*cols + j;
                _mm256_storeu_pd(data + offset, vals);
        }

    }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i<rows; i++) {
            for (int j = boundary; j<cols; j++) {
                    int offset = i * cols + j;
                    data[offset] = val;
            }
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int rows = result->rows;
    int cols = result->cols;
    double *A = mat->data;
    double *B = result->data;

    int boundary = cols/4*4;
    const __m256d b = _mm256_set1_pd(-1.0);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<boundary; j+= 4) {
                int offset = i * cols + j;
                __m256d a = _mm256_loadu_pd(A + offset);
                _mm256_storeu_pd(B + offset, _mm256_max_pd(_mm256_mul_pd(b, a), a));
        }
    }
    #pragma omp parallel for collapse(2)
    for (int i = 0; i<rows; i++) {
        for (int j = boundary; j<cols; j++) {
            int offset = i * cols + j;
            B[offset] = fabs(A[offset]);
        }
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int rows = result->rows;
    int cols = result->cols;
    double *A = mat1->data;
    double *B = mat2->data;
    double *C = result->data;
    int boundary = cols/4*4;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<boundary; j+= 4) {
            int offset = i * cols + j;
            __m256d a = _mm256_loadu_pd(A + offset);
            __m256d b = _mm256_loadu_pd(B + offset);
            _mm256_storeu_pd(C + offset, _mm256_add_pd(a, b));
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i<rows; i++) {
        for (int j = boundary; j<cols; j++) {
                int offset = i * cols + j;
                C[offset] = A[offset] + B[offset];
        }
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    return 0;
}

//Function to find the transpose of the matrix 
int transpose_matrix(matrix *mat, matrix* tempMatrix){
    # pragma omp parallel for
    for(int i = 0; i < mat->rows; i++){
        for(int j = 0; j < mat->cols/ 8 * 8; j+=8){
            tempMatrix->data[i + (j) *tempMatrix->cols] = mat->data[j + i * mat->cols];
            tempMatrix->data[i + (1 + j) * tempMatrix->cols] = mat->data[1 + j + i * mat->cols];
            tempMatrix->data[i + (2 + j) * tempMatrix->cols] = mat->data[2 + j + i * mat->cols];
            tempMatrix->data[i + (3 + j) * tempMatrix->cols] = mat->data[3 + j + i * mat->cols];
            tempMatrix->data[i + (4 + j) * tempMatrix->cols] = mat->data[4 + j + i * mat->cols];
            tempMatrix->data[i + (5 + j) * tempMatrix->cols] = mat->data[5 + j + i * mat->cols];
            tempMatrix->data[i + (6 + j) * tempMatrix->cols] = mat->data[6 + j + i * mat->cols];
            tempMatrix->data[i + (7 + j) * tempMatrix->cols] = mat->data[7 + j + i * mat->cols];
        }
        for(int j = mat->cols/ 8 * 8; j < mat->cols/ 4 * 4; j+=4){
            tempMatrix->data[i + (j) * tempMatrix->cols] = mat->data[j + i * mat->cols];
            tempMatrix->data[i + (1 + j) * tempMatrix->cols] = mat->data[1 + j + i * mat->cols];
            tempMatrix->data[i + (2 + j) * tempMatrix->cols] = mat->data[2 + j + i * mat->cols];
            tempMatrix->data[i + (3 + j) * tempMatrix->cols] = mat->data[3 + j + i * mat->cols];
        }
  
        for(int j =  mat->cols/ 4 * 4; j < mat->cols; j++){
            tempMatrix->data[i + j * tempMatrix->cols] = mat->data[j + i * mat->cols];
        }
    }
    
    return 0;
}
/*
/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
        
    // Task 1.6 TODO
    double* result_data = result->data;
    double* mat1_data = mat1->data;
    int col1 = mat1->cols;
    int col2 = mat2->cols;
    int row1 = mat1->rows;

    matrix * mat2_Transpose;
    allocate_matrix(&mat2_Transpose, mat2->cols, mat2->rows);
    transpose_matrix(mat2, mat2_Transpose);
    double* mat2_Transpose_data = mat2_Transpose->data;

    # pragma omp parallel for
    for(int i = 0; i < row1; i++){
        for(int j = 0; j < col2; j++){
                __m256d res = _mm256_set1_pd(0.0);
            if (col1 >= 4){ 
                for (int k = 0; k < col1 / 8 * 8; k += 8){
                    __m256d r1[2];
                    r1[0] = _mm256_loadu_pd(mat1_data + (i * col1) + k);
                    r1[1] = _mm256_loadu_pd(mat1_data + (i * col1) + k + 4);
                    __m256d c1[2];
                    c1[0] = _mm256_loadu_pd(mat2_Transpose_data + (j * col1) + k);
                    c1[1] = _mm256_loadu_pd(mat2_Transpose_data + (j * col1) + k + 4);
                    res = _mm256_fmadd_pd(r1[0], c1[0], res);
                    res = _mm256_fmadd_pd(r1[1], c1[1], res);
                }

                for (int k = col1 / 8 * 8; k < col1 / 4 * 4; k += 4){
                    __m256d r1 = _mm256_loadu_pd(mat1_data + (i * col1) + k);
                    __m256d c1 = _mm256_loadu_pd(mat2_Transpose_data + (j * col1) + k);
                    res = _mm256_fmadd_pd(r1, c1, res);
                }
                double tmp_arr[4];
                _mm256_storeu_pd(tmp_arr, res);
                result_data[col2 * i + j] = tmp_arr[0] + tmp_arr[1] + tmp_arr[2] + tmp_arr[3];
            }
            for(int k = col1 / 4 * 4; k < col1; k++){
                result_data[col2 * i + j] += mat1_data[i * col1 + k] * mat2_Transpose_data[j * col1 + k];
            }
        }

    }
    deallocate_matrix(mat2_Transpose);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    //I'm going to go for an iterative solution here
    //Note: if pow == 0, then curr_pow = 0 < 0 will always be false. 
    //Therefore, the inevitable idea is to intialize result to the identity matrix.
    if(result == NULL || mat == NULL) {
        return -2;
    }
        
    if (pow == 0) {
        for(int i = 0; i < result->rows; i++) {
            set(result, i, i, 1.0);
        }
        return 0;
    }

    if (pow == 1) {
        for(int i = 0; i < result->rows * result->cols; i++){
            memcpy(result->data, mat->data, mat->cols * mat->rows * sizeof(double));
            return 0;
        }
    }
        
    if (pow == 2) {
        mul_matrix(result, mat, mat);
        return 0;
    }
    pow_matrix(result, mat, pow/2);
    matrix *temp;
    allocate_matrix(&temp, result->rows, result->cols);
    mul_matrix(temp, result, result);
    free(result->data);
    result->data = temp->data;
    free (temp);

    if (pow % 2 != 0){
        matrix *dum;
        allocate_matrix(&dum, result->rows, result->cols);
        mul_matrix(dum, result, mat);
        free(result->data);
        result->data = dum->data;
        free (dum);
    }

    return 0;
}