'use strict';

import {buildConstantByNpy} from '../common/utils.js';

export class RNNoise {
  constructor(modelPath, batchSize, frames) {
    this.baseUrl_ = modelPath;
    this.batchSize_ = batchSize;
    this.frames_ = frames;
    this.model_ = null;
    this.graph_ = null;
    this.builder_ = null;
  }

  async load() {
    const context = navigator.ml.createContext();
    this.builder_ = new MLGraphBuilder(context);
    // Create constants by loading pre-trained data from .npy files.
    const inputDenseKernel0 = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'input_dense_kernel_0.npy');
    const inputDenseBias0 = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'input_dense_bias_0.npy');
    const vadGruW = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'vad_gru_W.npy');
    const vadGruR = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'vad_gru_R.npy');
    const vadGruBData = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'vad_gru_B.npy');
    const noiseGruW = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'noise_gru_W.npy');
    const noiseGruR = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'noise_gru_R.npy');
    const noiseGruBData = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'noise_gru_B.npy');
    const denoiseGruW = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'denoise_gru_W.npy');
    const denoiseGruR = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'denoise_gru_R.npy');
    const denoiseGruBData = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'denoise_gru_B.npy');
    const denoiseOutputKernel0 = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'denoise_output_kernel_0.npy');
    const denoiseOutputBias0 = await buildConstantByNpy(this.builder_,
      this.baseUrl_ + 'denoise_output_bias_0.npy');
    // Build up the network.
    const input = this.builder_.input(
      'input', {type: 'float32', dimensions: [this.batchSize_, this.frames_, 42]});
    const inputDense0 = this.builder_.matmul(input, inputDenseKernel0);
    const biasedTensorName2 = this.builder_.add(inputDense0, inputDenseBias0);
    const inputDenseTanh0 = this.builder_.tanh(biasedTensorName2);
    const vadGruX = this.builder_.transpose(
      inputDenseTanh0, {permutation: [1, 0, 2]});
    // **********hiddenSize = 24**********
    const vadGruB = this.builder_.slice(
      vadGruBData, [0], [3 * 24], {axes: [1]});
    const vadGruRB = this.builder_.slice(
      vadGruBData, [3 * 24], [-1], {axes: [1]});
    const vadGruInitialH = this.builder_.input('vadGruInitialH',
      {type: 'float32', dimensions: [1, this.batchSize_, 24]});
    const [vadGruYH, vadGruY] = this.builder_.gru(
      vadGruX, vadGruW, vadGruR, this.frames_, 24, {
        bias: vadGruB, recurrentBias: vadGruRB,
        initialHiddenState: vadGruInitialH,
        returnSequence: true, resetAfter: false,
        activations: ['sigmoid', 'relu'],
      });
    const vadGruYTransposed = this.builder_.transpose(
      vadGruY, {permutation: [2, 0, 1, 3]});
    const vadGruTranspose1 = this.builder_.reshape(
      vadGruYTransposed, [-1, this.frames_, 24]);
    const concatenate1 = this.builder_.concat(
      [inputDenseTanh0, vadGruTranspose1, input], 2);
    const noiseGruX = this.builder_.transpose(
      concatenate1, {permutation: [1, 0, 2]});
    // **********hiddenSize = 48**********
    const noiseGruB = this.builder_.slice(
      noiseGruBData, [0], [3 * 48], {axes: [1]});
    const noiseGruRB = this.builder_.slice(
      noiseGruBData, [3 * 48], [-1], {axes: [1]});
    const noiseGruInitialH = this.builder_.input( 'noiseGruInitialH',
      {type: 'float32', dimensions: [1, this.batchSize_, 48]});
    const [noiseGruYH, noiseGruY] = this.builder_.gru(
      noiseGruX, noiseGruW, noiseGruR, this.frames_, 48, {
        bias: noiseGruB, recurrentBias: noiseGruRB,
        initialHiddenState: noiseGruInitialH,
        returnSequence: true, resetAfter: false,
        activations: ['sigmoid', 'relu'],
      });
    const noiseGruYTransposed = this.builder_.transpose(
      noiseGruY, {permutation: [2, 0, 1, 3]});
    const noiseGruTranspose1 = this.builder_.reshape(
      noiseGruYTransposed, [-1, this.frames_, 48]);
    const concatenate2 = this.builder_.concat(
      [vadGruTranspose1, noiseGruTranspose1, input], 2);
    const denoiseGruX = this.builder_.transpose(
      concatenate2, {permutation: [1, 0, 2]});
    // **********hiddenSize = 96**********
    const denoiseGruB = this.builder_.slice(
      denoiseGruBData, [0], [3 * 96], {axes: [1]});
    const denoiseGruRB = this.builder_.slice(
      denoiseGruBData, [3 * 96], [-1], {axes: [1]});
    const denoiseGruInitialH = this.builder_.input('denoiseGruInitialH',
      {type: 'float32', dimensions: [1, this.batchSize_, 96]});
    const [denoiseGruYH, denoiseGruY] = this.builder_.gru(
      denoiseGruX, denoiseGruW, denoiseGruR, this.frames_, 96, {
        bias: denoiseGruB, recurrentBias: denoiseGruRB,
        initialHiddenState: denoiseGruInitialH,
        returnSequence: true, resetAfter: false,
        activations: ['sigmoid', 'relu'],
      });
    const denoiseGruYTransposed = this.builder_.transpose(
      denoiseGruY, {permutation: [2, 0, 1, 3]});
    const denoiseGruTranspose1 = this.builder_.reshape(
      denoiseGruYTransposed, [-1, this.frames_, 96]);
    const denoiseOutput0 = this.builder_.matmul(
      denoiseGruTranspose1, denoiseOutputKernel0);
    const biasedTensorName = this.builder_.add(
      denoiseOutput0, denoiseOutputBias0);
    const denoiseOutput = this.builder_.sigmoid(biasedTensorName);

    return {denoiseOutput, vadGruYH, noiseGruYH, denoiseGruYH};
  }

  build(outputOperand) {
    this.graph_ = this.builder_.build(outputOperand);
  }

  compute(inputBuffer, vadGruInitialHBuffer, noiseGruInitialHBuffer,
          denoiseGruInitialHBuffer, outputs) {
    const inputs = {
      'input': inputBuffer,
      'vadGruInitialH': vadGruInitialHBuffer,
      'noiseGruInitialH': noiseGruInitialHBuffer,
      'denoiseGruInitialH': denoiseGruInitialHBuffer,
    };
    this.graph_.compute(inputs, outputs);
  }
}
