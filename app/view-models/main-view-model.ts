import { Observable } from '@nativescript/core';
import { ModelManager, DownloadProgress } from '../models/model-manager';
import { ModelConfig, ModelFormat, ModelParameters } from '../models/types';

export class MainViewModel extends Observable {
    private modelManager: ModelManager;
    private _localModels: ModelConfig[];
    private _isLoading: boolean;
    private _downloadProgress: number;
    private _modelId: string;
    private _error: string;
    private _parameters: ModelParameters;
    private _selectedFormatIndex: number;
    private _selectedPrecisionIndex: number;

    readonly modelFormats: ModelFormat[] = ['pytorch', 'tensorflow', 'onnx', 'safetensors'];
    readonly precisionOptions = ['float32', 'float16', 'bfloat16'];

    constructor() {
        super();
        this.modelManager = new ModelManager();
        this._localModels = [];
        this._isLoading = false;
        this._downloadProgress = 0;
        this._modelId = '';
        this._error = '';
        this._selectedFormatIndex = 0;
        this._selectedPrecisionIndex = 1;
        this._parameters = this.getDefaultParameters();
        this.loadLocalModels();
    }

    private loadLocalModels() {
        this._localModels = this.modelManager.getLocalModels();
        this.notifyPropertyChange('localModels', this._localModels);
    }

    private getDefaultParameters(): ModelParameters {
        return this.modelManager['getDefaultParameters'](this.modelFormats[this._selectedFormatIndex]);
    }

    async downloadModel() {
        if (!this._modelId) {
            this._error = 'Please enter a model ID';
            this.notifyPropertyChange('error', this._error);
            return;
        }

        try {
            this._isLoading = true;
            this._error = '';
            this.notifyPropertyChange('isLoading', true);
            this.notifyPropertyChange('error', this._error);
            
            await this.modelManager.downloadModel(
                this._modelId,
                this.modelFormats[this._selectedFormatIndex],
                (progress: DownloadProgress) => {
                    this._downloadProgress = progress.percentage;
                    this.notifyPropertyChange('downloadProgress', this._downloadProgress);
                }
            );
            
            this.loadLocalModels();
            this._modelId = '';
            this.notifyPropertyChange('modelId', this._modelId);
            
        } catch (error) {
            this._error = error.message;
            this.notifyPropertyChange('error', this._error);
        } finally {
            this._isLoading = false;
            this._downloadProgress = 0;
            this.notifyPropertyChange('isLoading', false);
            this.notifyPropertyChange('downloadProgress', this._downloadProgress);
        }
    }

    async deleteModel(modelId: string) {
        try {
            await this.modelManager.deleteModel(modelId);
            this.loadLocalModels();
        } catch (error) {
            this._error = error.message;
            this.notifyPropertyChange('error', this._error);
        }
    }

    async saveParameters() {
        if (this._localModels.length === 0) {
            this._error = 'No model selected';
            this.notifyPropertyChange('error', this._error);
            return;
        }

        try {
            const selectedModel = this._localModels[0]; // For now, use first model
            await this.modelManager.updateModelParameters(selectedModel.id, this._parameters);
            this._error = 'Parameters saved successfully';
            this.notifyPropertyChange('error', this._error);
        } catch (error) {
            this._error = error.message;
            this.notifyPropertyChange('error', this._error);
        }
    }

    formatSize(bytes: number): string {
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;
        
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        
        return `${size.toFixed(2)} ${units[unitIndex]}`;
    }

    // Computed properties
    get showPyTorchParams(): boolean {
        return this.modelFormats[this._selectedFormatIndex] === 'pytorch';
    }

    get showTensorFlowParams(): boolean {
        return this.modelFormats[this._selectedFormatIndex] === 'tensorflow';
    }

    get showONNXParams(): boolean {
        return this.modelFormats[this._selectedFormatIndex] === 'onnx';
    }

    get showSafeTensorsParams(): boolean {
        return this.modelFormats[this._selectedFormatIndex] === 'safetensors';
    }

    // Getters and setters
    get localModels(): ModelConfig[] {
        return this._localModels;
    }

    get isLoading(): boolean {
        return this._isLoading;
    }

    get downloadProgress(): number {
        return this._downloadProgress;
    }

    get modelId(): string {
        return this._modelId;
    }

    set modelId(value: string) {
        if (this._modelId !== value) {
            this._modelId = value;
            this.notifyPropertyChange('modelId', value);
        }
    }

    get error(): string {
        return this._error;
    }

    get parameters(): ModelParameters {
        return this._parameters;
    }

    get selectedFormatIndex(): number {
        return this._selectedFormatIndex;
    }

    set selectedFormatIndex(value: number) {
        if (this._selectedFormatIndex !== value) {
            this._selectedFormatIndex = value;
            this._parameters = this.getDefaultParameters();
            this.notifyPropertyChange('selectedFormatIndex', value);
            this.notifyPropertyChange('parameters', this._parameters);
            this.notifyPropertyChange('showPyTorchParams', this.showPyTorchParams);
            this.notifyPropertyChange('showTensorFlowParams', this.showTensorFlowParams);
            this.notifyPropertyChange('showONNXParams', this.showONNXParams);
            this.notifyPropertyChange('showSafeTensorsParams', this.showSafeTensorsParams);
        }
    }

    get selectedPrecisionIndex(): number {
        return this._selectedPrecisionIndex;
    }

    set selectedPrecisionIndex(value: number) {
        if (this._selectedPrecisionIndex !== value) {
            this._selectedPrecisionIndex = value;
            if (this._parameters.safetensors) {
                this._parameters.safetensors.precision = this.precisionOptions[value] as any;
            }
            this.notifyPropertyChange('selectedPrecisionIndex', value);
            this.notifyPropertyChange('parameters', this._parameters);
        }
    }
}