import { sizeOfShape, createGPUBuffer, readbackGPUBuffer } from "./utils.js";

export class BaseNetwork {
  constructor() {
    this.device_ = null;
    this.builder_ = null;
    this.graph_ = null;
  }
  async init() {
    const adaptor = await navigator.gpu.requestAdapter();
    this.device_ = await adaptor.requestDevice();
    this.inputSizeInBytes_ = sizeOfShape(this.inputOptions.inputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.outputSizeInBytes_ = sizeOfShape(this.outputDimensions) * Float32Array.BYTES_PER_ELEMENT;
    this.inputGPUBuffers_ = [];
    this.outputGPUBuffer_ = null;
  }


  async build(outputOperand) {
    this.graph_ = this.builder_.build({'output': outputOperand});
  }

  async computeGPUTensor(inputData, outputBuffer, typedArrayConstructor = Float32Array) {
    const inputGPUBuffer = await createGPUBuffer(this.device_, sizeOfShape(this.inputOptions.inputDimensions), inputData);
    const outputGPUBuffer =
      // this.device_.createBuffer({size: this.outputSizeInBytes_, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
      // DF: models tests invoke below function 
      await createGPUBuffer(this.device_, sizeOfShape(this.outputDimensions));    
    this.graph_.compute({'input': {resource: inputGPUBuffer}}, {'output': {resource: outputGPUBuffer}});
    // await this.outputGPUBuffer_.mapAsync(GPUMapMode.READ);
    // outputBuffer.set(new typedArrayConstructor(this.outputGPUBuffer_.getMappedRange()));
    // this.outputGPUBuffer_.unmap();

    // DF: models tests invoke below function 
    const outputData = await readbackGPUBuffer(this.device_, sizeOfShape(this.outputDimensions), outputGPUBuffer);
    outputBuffer.set(outputData);
  }

  async compute(inputBuffer, outputBuffer, typedArrayConstructor = Float32Array) {
    let inputGPUBuffer;
    if (this.inputGPUBuffers_.length) {
      inputGPUBuffer = this.inputGPUBuffers_.pop();
    } else {
      console.log('create buffer');
      inputGPUBuffer = this.device_.createBuffer({
        size: this.inputSizeInBytes_,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC
      });
      await inputGPUBuffer.mapAsync(GPUMapMode.WRITE);
    }
    new Float32Array(inputGPUBuffer.getMappedRange()).set(inputBuffer);
    inputGPUBuffer.unmap();
    this.graph_.compute({'input': {resource: inputGPUBuffer}}, {'output': {resource: this.outputGPUBuffer_}});
    inputGPUBuffer.mapAsync(GPUMapMode.WRITE).then(() => {
      this.inputGPUBuffers_.push(inputGPUBuffer);
    });
    await this.outputGPUBuffer_.mapAsync(GPUMapMode.READ);
    outputBuffer.set(new typedArrayConstructor(this.outputGPUBuffer_.getMappedRange()));
    this.outputGPUBuffer_.unmap();
  }
}