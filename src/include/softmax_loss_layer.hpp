#ifndef CAFFE_AUMaskBasedLand_LAYER_HPP_
#define CAFFE_AUMaskBasedLand_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class AUMaskBasedLandLayer : public Layer<Dtype> {
	public:
		explicit AUMaskBasedLandLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const { return "AUMaskBasedLand"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 13; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int img_size_, mask_size_, AU_num_, num_, dim_, top_dim_;
		Dtype spatial_ratio_, spatial_scale_, half_AU_size_;
		Dtype fill_coeff_, fill_value_;
	};

}  // namespace caffe

#endif  // CAFFE_AUMaskBasedLand_LAYER_HPP_

#ifndef CAFFE_DICE_COEF_LOSS_LAYER_HPP_
#define CAFFE_DICE_COEF_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Euclidean (L2) loss @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$ for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E =
 *          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$
 *
 * This can be used for least-squares regression tasks.  An InnerProductLayer
 * input to a DiceCoefLossLayer exactly formulates a linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/caffe/test/test_sgd_solver.cpp for a concrete
 * example wherein we check that the gradients computed for a Net with exactly
 * this structure match hand-computed gradient formulas for ridge regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 */
template <typename Dtype>
class DiceCoefLossLayer : public LossLayer<Dtype> {
 public:
  explicit DiceCoefLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), multiplier_() ,result_() ,result_tmp_() ,tmp_(){}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DiceCoefLoss"; }
  /**
   * Unlike most loss layers, in the DiceCoefLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc DiceCoefLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Euclidean error gradient w.r.t. the inputs.
   *
   * Unlike other children of LossLayer, DiceCoefLossLayer \b can compute
   * gradients with respect to the label inputs bottom[1] (but still only will
   * if propagate_down[1] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> multiplier_;
  Blob<Dtype> result_;
  Blob<Dtype> result_tmp_;
  Blob<Dtype> tmp_;
  
  Dtype smooth;
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#ifndef CAFFE_DIVISION_LAYER_HPP_
#define CAFFE_DIVISION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Takes a Blob and spatially divide it into multiple divided Blob results.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class DivisionLayer : public Layer<Dtype> {
 public:
  explicit DivisionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Division"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int count_;
  vector <int> xcoord_;
  vector <int> ycoord_;
  int width_;
  int height_;
  int num_;
  int channels_;
};

}  // namespace caffe

#endif  // CAFFE_DIVISION_LAYER_HPP_

#ifndef CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Euclidean (L2) loss @f$
 *          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$ for real-valued regression tasks.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Euclidean loss: @f$ E =
 *          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
 *        \right| \right|_2^2 @f$
 *
 * This can be used for least-squares regression tasks.  An InnerProductLayer
 * input to a EuclideanLossLayer exactly formulates a linear least squares
 * regression problem. With non-zero weight decay the problem becomes one of
 * ridge regression -- see src/caffe/test/test_sgd_solver.cpp for a concrete
 * example wherein we check that the gradients computed for a Net with exactly
 * this structure match hand-computed gradient formulas for ridge regression.
 *
 * (Note: Caffe, and SGD in general, is certainly \b not the best way to solve
 * linear least squares problems! We use it only as an instructive example.)
 */
template <typename Dtype>
class EuclideanLossLayer : public LossLayer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "EuclideanLoss"; }
  /**
   * Unlike most loss layers, in the EuclideanLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc EuclideanLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the Euclidean error gradient w.r.t. the inputs.
   *
   * Unlike other children of LossLayer, EuclideanLossLayer \b can compute
   * gradients with respect to the label inputs bottom[1] (but still only will
   * if propagate_down[1] is set, due to being produced by learnable parameters
   * or if force_backward is set). In fact, this layer is "commutative" -- the
   * result is the same regardless of the order of the two bottoms.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$\hat{y}@f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial \hat{y}} =
   *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
   *      @f$ if propagate_down[0]
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets @f$y@f$; Backward fills their diff with gradients
   *      @f$ \frac{\partial E}{\partial y} =
   *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
   *      @f$ if propagate_down[1]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};

template <typename Dtype>
class EuclideanLoss2Layer : public LossLayer<Dtype> {
public:
	explicit EuclideanLoss2Layer(const LayerParameter& param)
		: LossLayer<Dtype>(param), diff_() {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "EuclideanLoss2"; }
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	/**
	* Unlike most loss layers, in the EuclideanLoss2Layer we can backpropagate
	* to both inputs -- override to return true and always allow force_backward.
	*/
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

protected:
	/// @copydoc EuclideanLoss2Layer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	/**
	* @brief Computes the Euclidean error gradient w.r.t. the inputs.
	*
	* Unlike other children of LossLayer, EuclideanLoss2Layer \b can compute
	* gradients with respect to the label inputs bottom[1] (but still only will
	* if propagate_down[1] is set, due to being produced by learnable parameters
	* or if force_backward is set). In fact, this layer is "commutative" -- the
	* result is the same regardless of the order of the two bottoms.
	*
	* @param top output Blob vector (length 1), providing the error gradient with
	*      respect to the outputs
	*   -# @f$ (1 \times 1 \times 1 \times 1) @f$
	*      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
	*      as @f$ \lambda @f$ is the coefficient of this layer's output
	*      @f$\ell_i@f$ in the overall Net loss
	*      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
	*      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
	*      (*Assuming that this top Blob is not used as a bottom (input) by any
	*      other layer of the Net.)
	* @param propagate_down see Layer::Backward.
	* @param bottom input Blob vector (length 2)
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the predictions @f$\hat{y}@f$; Backward fills their diff with
	*      gradients @f$
	*        \frac{\partial E}{\partial \hat{y}} =
	*            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
	*      @f$ if propagate_down[0]
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the targets @f$y@f$; Backward fills their diff with gradients
	*      @f$ \frac{\partial E}{\partial y} =
	*          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
	*      @f$ if propagate_down[1]
	*/
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> diff_;
};

template <typename Dtype>
class EuclideanLoss3Layer : public LossLayer<Dtype> {
public:
	explicit EuclideanLoss3Layer(const LayerParameter& param)
		: LossLayer<Dtype>(param), diff_() {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "EuclideanLoss3"; }
	virtual inline int ExactNumBottomBlobs() const { return 4; }
	/**
	* Unlike most loss layers, in the EuclideanLoss3Layer we can backpropagate
	* to both inputs -- override to return true and always allow force_backward.
	*/
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

protected:
	/// @copydoc EuclideanLoss3Layer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	/**
	* @brief Computes the Euclidean error gradient w.r.t. the inputs.
	*
	* Unlike other children of LossLayer, EuclideanLoss3Layer \b can compute
	* gradients with respect to the label inputs bottom[1] (but still only will
	* if propagate_down[1] is set, due to being produced by learnable parameters
	* or if force_backward is set). In fact, this layer is "commutative" -- the
	* result is the same regardless of the order of the two bottoms.
	*
	* @param top output Blob vector (length 1), providing the error gradient with
	*      respect to the outputs
	*   -# @f$ (1 \times 1 \times 1 \times 1) @f$
	*      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
	*      as @f$ \lambda @f$ is the coefficient of this layer's output
	*      @f$\ell_i@f$ in the overall Net loss
	*      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
	*      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
	*      (*Assuming that this top Blob is not used as a bottom (input) by any
	*      other layer of the Net.)
	* @param propagate_down see Layer::Backward.
	* @param bottom input Blob vector (length 2)
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the predictions @f$\hat{y}@f$; Backward fills their diff with
	*      gradients @f$
	*        \frac{\partial E}{\partial \hat{y}} =
	*            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
	*      @f$ if propagate_down[0]
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the targets @f$y@f$; Backward fills their diff with gradients
	*      @f$ \frac{\partial E}{\partial y} =
	*          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
	*      @f$ if propagate_down[1]
	*/
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> diff_;
};

template <typename Dtype>
class EuclideanLoss4Layer : public LossLayer<Dtype> {
public:
	explicit EuclideanLoss4Layer(const LayerParameter& param)
		: LossLayer<Dtype>(param), diff_() {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "EuclideanLoss4"; }
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	/**
	* Unlike most loss layers, in the EuclideanLoss4Layer we can backpropagate
	* to both inputs -- override to return true and always allow force_backward.
	*/
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

protected:
	/// @copydoc EuclideanLoss4Layer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	/**
	* @brief Computes the Euclidean error gradient w.r.t. the inputs.
	*
	* Unlike other children of LossLayer, EuclideanLoss4Layer \b can compute
	* gradients with respect to the label inputs bottom[1] (but still only will
	* if propagate_down[1] is set, due to being produced by learnable parameters
	* or if force_backward is set). In fact, this layer is "commutative" -- the
	* result is the same regardless of the order of the two bottoms.
	*
	* @param top output Blob vector (length 1), providing the error gradient with
	*      respect to the outputs
	*   -# @f$ (1 \times 1 \times 1 \times 1) @f$
	*      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
	*      as @f$ \lambda @f$ is the coefficient of this layer's output
	*      @f$\ell_i@f$ in the overall Net loss
	*      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
	*      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
	*      (*Assuming that this top Blob is not used as a bottom (input) by any
	*      other layer of the Net.)
	* @param propagate_down see Layer::Backward.
	* @param bottom input Blob vector (length 2)
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the predictions @f$\hat{y}@f$; Backward fills their diff with
	*      gradients @f$
	*        \frac{\partial E}{\partial \hat{y}} =
	*            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
	*      @f$ if propagate_down[0]
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the targets @f$y@f$; Backward fills their diff with gradients
	*      @f$ \frac{\partial E}{\partial y} =
	*          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
	*      @f$ if propagate_down[1]
	*/
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> diff_;
};

template <typename Dtype>
class EuclideanLoss5Layer : public LossLayer<Dtype> {
public:
	explicit EuclideanLoss5Layer(const LayerParameter& param)
		: LossLayer<Dtype>(param), diff_() {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "EuclideanLoss5"; }
	virtual inline int ExactNumBottomBlobs() const { return 3; }
	/**
	* Unlike most loss layers, in the EuclideanLoss4Layer we can backpropagate
	* to both inputs -- override to return true and always allow force_backward.
	*/
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

protected:
	/// @copydoc EuclideanLoss5Layer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	/**
	* @brief Computes the Euclidean error gradient w.r.t. the inputs.
	*
	* Unlike other children of LossLayer, EuclideanLoss4Layer \b can compute
	* gradients with respect to the label inputs bottom[1] (but still only will
	* if propagate_down[1] is set, due to being produced by learnable parameters
	* or if force_backward is set). In fact, this layer is "commutative" -- the
	* result is the same regardless of the order of the two bottoms.
	*
	* @param top output Blob vector (length 1), providing the error gradient with
	*      respect to the outputs
	*   -# @f$ (1 \times 1 \times 1 \times 1) @f$
	*      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
	*      as @f$ \lambda @f$ is the coefficient of this layer's output
	*      @f$\ell_i@f$ in the overall Net loss
	*      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
	*      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
	*      (*Assuming that this top Blob is not used as a bottom (input) by any
	*      other layer of the Net.)
	* @param propagate_down see Layer::Backward.
	* @param bottom input Blob vector (length 2)
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the predictions @f$\hat{y}@f$; Backward fills their diff with
	*      gradients @f$
	*        \frac{\partial E}{\partial \hat{y}} =
	*            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
	*      @f$ if propagate_down[0]
	*   -# @f$ (N \times C \times H \times W) @f$
	*      the targets @f$y@f$; Backward fills their diff with gradients
	*      @f$ \frac{\partial E}{\partial y} =
	*          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
	*      @f$ if propagate_down[1]
	*/
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> diff_;
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_

#ifndef CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief Computes the multinomial logistic loss for a one-of-many
 *        classification task, passing real-valued predictions through a
 *        softmax to get a probability distribution over classes.
 *
 * This layer should be preferred over separate
 * SoftmaxLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SoftmaxLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ x @f$, a Blob with values in
 *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
 *      the @f$ K = CHW @f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      @f$ \hat{p}_{nk} = \exp(x_{nk}) /
 *      \left[\sum_{k'} \exp(x_{nk'})\right] @f$ (see SoftmaxLayer).
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
 *      indicating the correct class label among the @f$ K @f$ classes
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed cross-entropy classification loss: @f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
 *      @f$, for softmax output class probabilites @f$ \hat{p} @f$
 */
template <typename Dtype>
class SoftmaxWithLossLayer : public LossLayer<Dtype> {
 public:
   /**
    * @param param provides LossParameter loss_param, with options:
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    *  - normalize (optional, default true)
    *    If true, the loss is normalized by the number of (nonignored) labels
    *    present; otherwise the loss is simply summed over spatial locations.
    */
  explicit SoftmaxWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /**
   * @brief Computes the softmax loss error gradient w.r.t. the predictions.
   *
   * Gradients cannot be computed with respect to the label inputs (bottom[1]),
   * so this method ignores bottom[1] and requires !propagate_down[1], crashing
   * if propagate_down[1] is set.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   *      propagate_down[1] must be false as we can't compute gradients with
   *      respect to the labels.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial x} @f$
   *   -# @f$ (N \times 1 \times 1 \times 1) @f$
   *      the labels -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// How to normalize the output loss.
  LossParameter_NormalizationMode normalization_;

  int softmax_axis_, outer_num_, inner_num_;
  const Dtype* weights_;
  bool has_weights_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
