import { File, Folder, knownFolders, path, Http } from '@nativescript/core';
import { ModelStorageError } from './model-storage-error';
import { ModelConfig, ModelFormat, ModelParameters } from './types';

export interface DownloadProgress {
    bytesReceived: number;
    totalBytes: number;
    percentage: number;
}

export class ModelManager {
    private readonly cacheFolder: Folder;
    private readonly configFile: string;
    private modelConfigs: Map<string, ModelConfig>;
    
    constructor() {
        this.cacheFolder = knownFolders.documents().getFolder('model-cache');
        this.configFile = path.join(this.cacheFolder.path, 'model-configs.json');
        this.modelConfigs = new Map();
        this.loadConfigs();
    }

    private loadConfigs() {
        if (File.exists(this.configFile)) {
            const content = File.fromPath(this.configFile).readTextSync();
            const configs = JSON.parse(content);
            this.modelConfigs = new Map(Object.entries(configs));
        }
    }

    private saveConfigs() {
        const configs = Object.fromEntries(this.modelConfigs);
        const file = File.fromPath(this.configFile);
        file.writeTextSync(JSON.stringify(configs, null, 2));
    }

    async downloadModel(
        modelId: string,
        format: ModelFormat = 'pytorch',
        onProgress?: (progress: DownloadProgress) => void
    ): Promise<ModelConfig> {
        const modelDir = path.join(this.cacheFolder.path, this.sanitizeModelId(modelId));
        
        if (this.modelConfigs.has(modelId)) {
            return this.modelConfigs.get(modelId);
        }

        try {
            if (!Folder.exists(modelDir)) {
                Folder.fromPath(modelDir);
            }

            const modelInfo = await this.fetchModelInfo(modelId);
            const files = await this.downloadModelFiles(modelId, modelDir, format, onProgress);
            
            const config: ModelConfig = {
                id: modelId,
                name: modelInfo.name,
                path: modelDir,
                files: files,
                size: this.calculateTotalSize(files),
                downloadDate: new Date().toISOString(),
                lastUsed: new Date().toISOString(),
                format: format,
                parameters: this.getDefaultParameters(format)
            };

            this.modelConfigs.set(modelId, config);
            this.saveConfigs();

            return config;
        } catch (error) {
            if (Folder.exists(modelDir)) {
                Folder.fromPath(modelDir).remove();
            }
            throw new ModelStorageError(`Failed to download model ${modelId}: ${error.message}`);
        }
    }

    private getDefaultParameters(format: ModelFormat): ModelParameters {
        const baseParams: ModelParameters = {
            temperature: 0.7,
            maxLength: 100,
            topK: 50,
            topP: 0.9,
            repetitionPenalty: 1.0,
            presencePenalty: 0.0,
            frequencyPenalty: 0.0,
            numBeams: 1,
            lengthPenalty: 1.0,
            doSample: true,
            earlyStopping: false
        };

        switch (format) {
            case 'pytorch':
                return {
                    ...baseParams,
                    pytorch: {
                        useHalfPrecision: true,
                        useFP16: true,
                        useQuantization: false,
                        quantizationBits: 16,
                        deviceMap: 'cpu',
                        torchDtype: 'float16'
                    }
                };
            case 'tensorflow':
                return {
                    ...baseParams,
                    tensorflow: {
                        useXLA: true,
                        useGPU: true,
                        mixedPrecision: true,
                        tensorflowVersion: '2.x'
                    }
                };
            case 'onnx':
                return {
                    ...baseParams,
                    onnx: {
                        optimizationLevel: 3,
                        graphOptimization: true,
                        parallelExecution: true,
                        enableMemoryOptimization: true
                    }
                };
            case 'safetensors':
                return {
                    ...baseParams,
                    safetensors: {
                        precision: 'float16',
                        useQuantization: false,
                        quantizationBits: 16
                    }
                };
            default:
                return baseParams;
        }
    }

    private async fetchModelInfo(modelId: string): Promise<any> {
        const response = await fetch(`https://huggingface.co/api/models/${modelId}`);
        if (!response.ok) {
            throw new ModelStorageError(`Model ${modelId} not found`);
        }
        return response.json();
    }

    private getModelFiles(format: ModelFormat): string[] {
        switch (format) {
            case 'pytorch':
                return ['config.json', 'pytorch_model.bin', 'tokenizer.json'];
            case 'tensorflow':
                return ['config.json', 'tf_model.h5', 'tokenizer.json'];
            case 'onnx':
                return ['config.json', 'model.onnx', 'tokenizer.json'];
            case 'safetensors':
                return ['config.json', 'model.safetensors', 'tokenizer.json'];
            default:
                throw new ModelStorageError(`Unsupported model format: ${format}`);
        }
    }

    private async downloadModelFiles(
        modelId: string,
        modelDir: string,
        format: ModelFormat,
        onProgress?: (progress: DownloadProgress) => void
    ): Promise<ModelFile[]> {
        const files: ModelFile[] = [];
        const modelFiles = this.getModelFiles(format);
        
        for (const filename of modelFiles) {
            const url = `https://huggingface.co/${modelId}/resolve/main/${filename}`;
            const filePath = path.join(modelDir, filename);
            
            try {
                await this.downloadFile(url, filePath, onProgress);
                const fileStats = File.fromPath(filePath);
                
                files.push({
                    name: filename,
                    path: filePath,
                    size: fileStats.size
                });
            } catch (error) {
                throw new ModelStorageError(`Failed to download ${filename}: ${error.message}`);
            }
        }
        
        return files;
    }

    private async downloadFile(
        url: string,
        filePath: string,
        onProgress?: (progress: DownloadProgress) => void
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            Http.getFile(url, filePath, {
                onProgress: (current, total) => {
                    if (onProgress) {
                        onProgress({
                            bytesReceived: current,
                            totalBytes: total,
                            percentage: (current / total) * 100
                        });
                    }
                }
            }).then(resolve).catch(reject);
        });
    }

    getLocalModels(): ModelConfig[] {
        return Array.from(this.modelConfigs.values());
    }

    async deleteModel(modelId: string): Promise<void> {
        const config = this.modelConfigs.get(modelId);
        if (!config) {
            throw new ModelStorageError(`Model ${modelId} not found`);
        }

        try {
            if (Folder.exists(config.path)) {
                Folder.fromPath(config.path).remove();
            }
            this.modelConfigs.delete(modelId);
            this.saveConfigs();
        } catch (error) {
            throw new ModelStorageError(`Failed to delete model ${modelId}: ${error.message}`);
        }
    }

    async updateModelParameters(modelId: string, parameters: Partial<ModelParameters>): Promise<void> {
        const config = this.modelConfigs.get(modelId);
        if (!config) {
            throw new ModelStorageError(`Model ${modelId} not found`);
        }

        config.parameters = { ...config.parameters, ...parameters };
        config.lastUsed = new Date().toISOString();
        this.saveConfigs();
    }

    private calculateTotalSize(files: ModelFile[]): number {
        return files.reduce((total, file) => total + file.size, 0);
    }

    private sanitizeModelId(modelId: string): string {
        return modelId.replace(/[^a-zA-Z0-9-_]/g, '_');
    }
}