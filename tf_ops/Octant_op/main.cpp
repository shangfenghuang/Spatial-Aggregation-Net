#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;


REGISTER_OP("CubeSelect")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Input("xyz_new: float32")
        .Output("idx: int32");
        
REGISTER_OP("CubeSelectTwo")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Input("xyz_new: float32")
        .Output("idx: int32");
        

REGISTER_OP("CubeSelectFour")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Input("xyz_new: float32")
        .Output("idx: int32");


REGISTER_OP("CubeSelectEight")
        .Attr("radius: float")
        .Input("xyz: float32")
        .Input("xyz_new: float32")
        .Output("idx: int32");
        




void cubeSelectLauncher(int b, int n,int n_new, float radius, const float* xyz,const float* xyz_new , int* idx_out);
class CubeSelectOp : public OpKernel {
public:
    explicit CubeSelectOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        const Tensor& xyz_new_tensor = context->input(1);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);
        int n_new = xyz_new_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 8}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto xyz_flat_new = xyz_new_tensor.flat<float>();
        const float* xyz_new = &(xyz_flat_new(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectLauncher(b, n, n_new, radius_, xyz, xyz_new, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelect").Device(DEVICE_GPU), CubeSelectOp);

void cubeSelectTwoLauncher(int b, int n,int n_new, float radius, const float* xyz,const float* xyz_new , int* idx_out);
class CubeSelectTwoOp : public OpKernel {
public:
    explicit CubeSelectTwoOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        const Tensor& xyz_new_tensor = context->input(1);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);
        int n_new = xyz_new_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 16}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto xyz_flat_new = xyz_new_tensor.flat<float>();
        const float* xyz_new = &(xyz_flat_new(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectTwoLauncher(b, n, n_new, radius_, xyz, xyz_new, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelectTwo").Device(DEVICE_GPU), CubeSelectTwoOp);

void cubeSelectFourLauncher(int b, int n,int n_new, float radius, const float* xyz,const float* xyz_new , int* idx_out);
class CubeSelectFourOp : public OpKernel {
public:
    explicit CubeSelectFourOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        const Tensor& xyz_new_tensor = context->input(1);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);
        int n_new = xyz_new_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 32}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto xyz_flat_new = xyz_new_tensor.flat<float>();
        const float* xyz_new = &(xyz_flat_new(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectFourLauncher(b, n, n_new, radius_, xyz, xyz_new, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelectFour").Device(DEVICE_GPU), CubeSelectFourOp);


void cubeSelectEightLauncher(int b, int n,int n_new, float radius, const float* xyz,const float* xyz_new , int* idx_out);
class CubeSelectEightOp : public OpKernel {
public:
    explicit CubeSelectEightOp(OpKernelConstruction * context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& xyz_tensor = context->input(0);
        const Tensor& xyz_new_tensor = context->input(1);
        int b = xyz_tensor.shape().dim_size(0);
        int n = xyz_tensor.shape().dim_size(1);
        int n_new = xyz_new_tensor.shape().dim_size(1);

        Tensor* idx_out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b, n, 64}, &idx_out_tensor));
        auto xyz_flat = xyz_tensor.flat<float>();
        const float* xyz = &(xyz_flat(0));

        auto xyz_flat_new = xyz_new_tensor.flat<float>();
        const float* xyz_new = &(xyz_flat_new(0));

        auto idx_out_flat = idx_out_tensor->flat<int>();
        int* idx_out = &(idx_out_flat(0));
        cubeSelectEightLauncher(b, n, n_new, radius_, xyz, xyz_new, idx_out);
    }
private:
    float radius_;
};
REGISTER_KERNEL_BUILDER(Name("CubeSelectEight").Device(DEVICE_GPU), CubeSelectEightOp);