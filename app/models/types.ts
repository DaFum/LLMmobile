export interface ModelFile {
    name: string;
    path: string;
    size: number;
}

export interface ModelConfig {
    id: string;
    name: string;
    path: string;
    files: ModelFile[];
    size: number;
    downloadDate: string;
    lastUsed: string;
    format: ModelFormat;
    parameters: ModelParameters;
}

export type ModelFormat = 'pytorch' | 'tensorflow' | 'onnx' | 'safetensors';

export interface ModelParameters {
    // Common parameters
    temperature: number;
    maxLength: number;
    topK: number;
    topP: number;
    repetitionPenalty: number;
    presencePenalty: number;
    frequencyPenalty: number;
    numBeams: number;
    lengthPenalty: number;
    doSample: boolean;
    earlyStopping: boolean;
    
    // Format specific parameters
    pytorch?: PyTorchParameters;
    tensorflow?: TensorFlowParameters;
    onnx?: ONNXParameters;
    safetensors?: SafeTensorsParameters;
}

export interface PyTorchParameters {
    useHalfPrecision: boolean;
    useFP16: boolean;
    useQuantization: boolean;
    quantizationBits: 8 | 16 | 32;
    deviceMap: 'cpu' | 'cuda' | 'mps';
    torchDtype: string;
}

export interface TensorFlowParameters {
    useXLA: boolean;
    useGPU: boolean;
    mixedPrecision: boolean;
    tensorflowVersion: string;
}

export interface ONNXParameters {
    optimizationLevel: number;
    graphOptimization: boolean;
    parallelExecution: boolean;
    enableMemoryOptimization: boolean;
}

export interface SafeTensorsParameters {
    precision: 'float32' | 'float16' | 'bfloat16';
    useQuantization: boolean;
    quantizationBits: 8 | 16 | 32;
}