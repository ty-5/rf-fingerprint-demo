import React, { useState, useEffect } from 'react';
import { Shuffle, Play, Eye, EyeOff, Layers, Zap, Clock } from 'lucide-react';
//import FingerprintComparison from './FingerprintComparison';

const SignalShuffler = () => {
    const [signalData, setSignalData] = useState(null);
    const [currentSignal, setCurrentSignal] = useState(null);
    const [showSignalId, setShowSignalId] = useState(false);

    // Classification state
    const [isClassifying, setIsClassifying] = useState(false);
    const [finalResult, setFinalResult] = useState(null);

    // Live visualization state
    const [isVisualizing, setIsVisualizing] = useState(false);
    const [layerData, setLayerData] = useState([]);
    const [currentLayer, setCurrentLayer] = useState(-1);
    const [processingStage, setProcessingStage] = useState('idle'); // 'idle', 'classifying', 'visualizing', 'complete'

    // Load signal data
    useEffect(() => {
        const loadSignalData = async () => {
            try {
                console.log('Loading signal data...');
                const response = await fetch('/rf_signals.json');
                if (!response.ok) throw new Error('Failed to load signal data');

                const data = await response.json();
                setSignalData(data);
                console.log(`Loaded ${Object.keys(data).length} transmitters`);
            } catch (error) {
                console.error('Failed to load signal data:', error);
            }


        };

        loadSignalData();
    }, []);

    // Get random signal
    const getRandomSignal = () => {
        if (!signalData) return;

        const transmitterKeys = Object.keys(signalData);
        const randomTransmitterKey = transmitterKeys[Math.floor(Math.random() * transmitterKeys.length)];

        const transmitterSamples = signalData[randomTransmitterKey];
        const randomSampleIndex = Math.floor(Math.random() * transmitterSamples.length);
        const selectedSample = transmitterSamples[randomSampleIndex];

        console.log(`Selected transmitter ${randomTransmitterKey}, sample ${randomSampleIndex}`);

        setCurrentSignal({
            signal: selectedSample,
            signalId: randomTransmitterKey,
            sampleIndex: randomSampleIndex
        });

        // Reset all states
        setFinalResult(null);
        setShowSignalId(false);
        setLayerData([]);
        setCurrentLayer(-1);
        setProcessingStage('idle');
    };

    // Process signal to correct format
    const prepareSignal = (signal) => {
        let processedSignal = signal;

        if (processedSignal[0]?.length !== 128) {
            processedSignal = processedSignal.map(channel => {
                if (channel.length < 128) {
                    const padding = new Array(128 - channel.length).fill(0);
                    return [...channel, ...padding];
                } else if (channel.length > 128) {
                    return channel.slice(0, 128);
                }
                return channel;
            });
        }

        return processedSignal.map(channel => channel.map(val => parseFloat(val)));
    };

    // Main classification with live visualization
    const classifyWithVisualization = async () => {
        if (!currentSignal) return;

        setIsClassifying(true);
        setIsVisualizing(true);
        setProcessingStage('classifying');
        setFinalResult(null);
        setLayerData([]);
        setCurrentLayer(-1);

        const processedSignal = prepareSignal(currentSignal.signal);

        try {
            // Step 1: Get final classification result (fast)
            console.log('Getting final classification...');
            const requestData = {
                signal: processedSignal,
                get_layers: false,
                metadata: {
                    true_transmitter: currentSignal.signalId,
                    sample_index: currentSignal.sampleIndex
                }
            };

            const response = await fetch('http://localhost:8000/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (response.ok) {
                const result = await response.json();
                setFinalResult(result);
                console.log('Final classification:', result.predicted_transmitter);
            }

            setIsClassifying(false);
            setProcessingStage('visualizing');

            // Step 2: Animate through layers (slower, for show)
            await animateLayerProgression(processedSignal);

        } catch (error) {
            console.error('Classification failed:', error);
            setIsClassifying(false);
            setIsVisualizing(false);
            setProcessingStage('idle');
        }
    };

    // Animate through CNN layers with proper timing
    const animateLayerProgression = async (processedSignal) => {
        const layerNames = ["Input", "ConvBlock1", "ConvBlock2", "ConvBlock3", "ConvBlock4", "Classifier"];
        const layerDescriptions = [
            "Raw I/Q Signal Input",
            "First Convolution Block - Basic Feature Detection",
            "Second Convolution Block - Pattern Recognition",
            "Third Convolution Block - Complex Features",
            "Fourth Convolution Block - High-Level Abstractions",
            "Final Classification Layer - Transmitter Identification"
        ];

        // Professional timing: slower for better comprehension
        const layerDelays = [800, 1200, 1200, 1200, 1200, 1000]; // milliseconds

        for (let layerIndex = 0; layerIndex < layerNames.length; layerIndex++) {
            try {
                console.log(`Processing layer ${layerIndex}: ${layerNames[layerIndex]}`);
                setCurrentLayer(layerIndex);

                const response = await fetch('http://localhost:8000/classify/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        signal: processedSignal,
                        layer_index: layerIndex
                    })
                });

                if (!response.ok) {
                    throw new Error(`Layer ${layerIndex} processing failed`);
                }

                const layerResult = await response.json();

                // Add description for better UX
                layerResult.description = layerDescriptions[layerIndex];

                // Update layer data progressively
                setLayerData(prev => [...prev, layerResult]);

                // Professional pacing - wait before next layer
                await new Promise(resolve => setTimeout(resolve, layerDelays[layerIndex]));

            } catch (error) {
                console.error(`Layer ${layerIndex} failed:`, error);
                break;
            }
        }

        setIsVisualizing(false);
        setProcessingStage('complete');
        console.log('Visualization complete!');
    };

    // Quick classification without visualization
    const quickClassify = async () => {
        if (!currentSignal) return;

        setIsClassifying(true);
        setProcessingStage('classifying');

        const processedSignal = prepareSignal(currentSignal.signal);

        try {
            const requestData = {
                signal: processedSignal,
                get_layers: false,
                metadata: {
                    true_transmitter: currentSignal.signalId,
                    sample_index: currentSignal.sampleIndex
                }
            };

            const response = await fetch('http://localhost:8000/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Classification failed: ${response.status}`);
            }

            const result = await response.json();
            setFinalResult(result);
            setProcessingStage('complete');
            console.log('Quick classification complete:', result);

        } catch (error) {
            console.error('Classification failed:', error);
        } finally {
            setIsClassifying(false);
        }
    };

    // Waveform visualization component
    const WaveformChart = ({ data, title, color, height = 100 }) => {
        if (!data || !Array.isArray(data)) return null;

        const width = 280;
        const padding = 15;
        const innerWidth = width - padding * 2;
        const innerHeight = height - padding * 2;

        const minVal = Math.min(...data);
        const maxVal = Math.max(...data);
        const range = maxVal - minVal || 1;

        const points = data.slice(0, Math.min(data.length, 128)).map((val, idx) => {
            const x = padding + (idx / (Math.min(data.length, 128) - 1)) * innerWidth;
            const y = padding + ((maxVal - val) / range) * innerHeight;
            return `${x},${y}`;
        }).join(' ');

        return (
            <div className="text-center">
                <div className="text-xs font-medium text-gray-700 mb-2">{title}</div>
                <svg width={width} height={height} className="border bg-gradient-to-b from-gray-50 to-white rounded">
                    <defs>
                        <linearGradient id={`grad-${title}`} x1="0%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" style={{ stopColor: color, stopOpacity: 0.3 }} />
                            <stop offset="100%" style={{ stopColor: color, stopOpacity: 0.1 }} />
                        </linearGradient>
                    </defs>
                    <polyline
                        points={points}
                        fill="none"
                        stroke={color}
                        strokeWidth="2"
                    />
                </svg>
                <div className="text-xs text-gray-500 mt-1">
                    Range: [{minVal.toFixed(3)}, {maxVal.toFixed(3)}]
                </div>
            </div>
        );
    };

    // Render layer visualization
    const renderLayerVisualization = (layer, index) => {
        const colors = ['#EF4444', '#F97316', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6'];
        const color = colors[index % colors.length];

        return (
            <div
                key={index}
                className={`p-4 bg-white rounded-lg border-2 transition-all duration-500 ${index === currentLayer ? 'border-purple-400 shadow-lg scale-105' : 'border-gray-200'
                    }`}
            >
                <div className="flex items-center gap-2 mb-3">
                    <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: color }}
                    ></div>
                    <h4 className="font-semibold text-sm">{layer.layer_name}</h4>
                    {index === currentLayer && (
                        <Zap className="text-purple-500 ml-auto animate-pulse" size={16} />
                    )}
                </div>

                <p className="text-xs text-gray-600 mb-3">{layer.description}</p>

                {/* Input layer: Show I/Q channels */}
                {layer.layer_name === "Input" && layer.feature_maps && (
                    <div className="space-y-3">
                        <WaveformChart
                            data={layer.feature_maps[0]}
                            title="I Channel (In-Phase)"
                            color="#EF4444"
                            height={80}
                        />
                        <WaveformChart
                            data={layer.feature_maps[1]}
                            title="Q Channel (Quadrature)"
                            color="#10B981"
                            height={80}
                        />
                    </div>
                )}

                {/* Convolutional layers: Show feature maps */}
                {layer.layer_name.includes("ConvBlock") && layer.feature_maps && (
                    <div className="grid grid-cols-2 gap-2">
                        {layer.feature_maps.slice(0, 4).map((featureMap, fIdx) => (
                            <WaveformChart
                                key={fIdx}
                                data={featureMap}
                                title={`Feature ${fIdx + 1}`}
                                color={color}
                                height={60}
                            />
                        ))}
                    </div>
                )}

                {/* Final layer: Show prediction probabilities */}
                {layer.layer_name === "Classifier" && layer.current_predictions && (
                    <div className="mt-3">
                        <div className="text-xs font-semibold mb-2">Transmitter Probabilities:</div>
                        <div className="grid grid-cols-4 gap-1 text-xs">
                            {layer.current_predictions.map((prob, pidx) => (
                                <div
                                    key={pidx}
                                    className={`text-center p-1 rounded ${prob > 0.1 ? 'bg-green-100 font-bold text-green-700' : 'text-gray-500'
                                        }`}
                                >
                                    T{pidx}: {(prob * 100).toFixed(0)}%
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Layer info */}
                {layer.layer_info && (
                    <div className="mt-2 text-xs text-gray-500">
                        {layer.layer_info.total_channels && (
                            <span>Channels: {layer.layer_info.total_channels} | </span>
                        )}
                        {layer.layer_info.time_samples && (
                            <span>Samples: {layer.layer_info.time_samples}</span>
                        )}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="unified-demo p-6 max-w-7xl mx-auto">
            {/* Header */}
            <div className="text-center mb-8">
                <h1 className="text-4xl font-bold mb-3">
                    RF Signal Classification Demo
                </h1>
                <p className="text-gray-600">
                </p>
            </div>

            {/* Status Bar */}
            <div className="mb-6 p-4 bg-gray-100 rounded-lg flex items-center justify-between">
                <div>
                    {signalData ? (
                        <p className="text-green-600">
                            Loaded {Object.keys(signalData).length} RF signals ready for classification
                        </p>
                    ) : (
                        <p className="text-gray-600">📡 Loading signal database...</p>
                    )}
                </div>

                {processingStage !== 'idle' && (
                    <div className="flex items-center gap-2 text-sm text-purple-600">
                        <Clock size={16} className="animate-spin" />
                        {processingStage === 'classifying' && 'Classifying signal...'}
                        {processingStage === 'visualizing' && 'Visualizing CNN layers...'}
                        {processingStage === 'complete' && 'Analysis complete!'}
                    </div>
                )}
            </div>

            {/* Main Demo Area */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Panel: Controls & Results */}
                <div className="lg:col-span-1 space-y-6">

                    {/* Signal Selection */}
                    <div className="bg-blue-50 p-6 rounded-lg">
                        <h3 className="font-semibold mb-4">Signal Selection</h3>
                        <button
                            onClick={getRandomSignal}
                            disabled={!signalData || isClassifying || isVisualizing}
                            className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                        >
                            <Shuffle size={20} />
                            Get Random Signal
                        </button>
                    </div>

                    {/* Current Signal Info */}
                    {currentSignal && (
                        <div className="bg-white p-6 rounded-lg border">
                            <h3 className="font-semibold mb-3">Current Signal</h3>
                            <div className="space-y-2 text-sm">
                                <p><span className="font-medium">Shape:</span> {currentSignal.signal.length} × {currentSignal.signal[0]?.length}</p>
                                <p><span className="font-medium">Sample:</span> #{currentSignal.sampleIndex}</p>
                                {showSignalId && (
                                    <p className="font-mono text-blue-600">
                                        <span className="font-medium">True Transmitter:</span> #{currentSignal.signalId}
                                    </p>
                                )}
                            </div>

                            {/* Classification Buttons */}
                            <div className="mt-4 space-y-2">
                                <button
                                    onClick={quickClassify}
                                    disabled={isClassifying || isVisualizing}
                                    className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition-colors"
                                >
                                    <Play size={16} />
                                    {isClassifying && !isVisualizing ? 'Classifying...' : 'Quick Classify'}
                                </button>

                                <button
                                    onClick={classifyWithVisualization}
                                    disabled={isClassifying || isVisualizing}
                                    className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-400 transition-colors"
                                >
                                    <Layers size={16} />
                                    {isVisualizing ? 'Visualizing...' : 'Classify with Live CNN View'}
                                </button>

                                {finalResult && (
                                    <button
                                        onClick={() => setShowSignalId(!showSignalId)}
                                        className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
                                    >
                                        {showSignalId ? <EyeOff size={16} /> : <Eye size={16} />}
                                        {showSignalId ? 'Hide Truth' : 'Reveal Truth'}
                                    </button>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Results */}
                    {finalResult && (
                        <div className="bg-green-50 p-6 rounded-lg border border-green-200">
                            <h3 className="font-semibold mb-3">Classification Results</h3>
                            <div className="space-y-3">
                                <div>
                                    <p className="text-sm text-gray-600">Predicted Transmitter:</p>
                                    <p className="text-2xl font-bold text-green-600">
                                        Transmitter {finalResult.predicted_transmitter}
                                    </p>
                                </div>
                                <div>
                                    <p className="text-sm text-gray-600">Confidence:</p>
                                    <p className="text-xl font-bold text-green-600">
                                        {(finalResult.confidence * 100).toFixed(1)}%
                                    </p>
                                </div>

                                {showSignalId && (
                                    <div className={`p-3 rounded ${finalResult.predicted_transmitter === parseInt(currentSignal.signalId)
                                            ? 'bg-green-100 text-green-800'
                                            : 'bg-red-100 text-red-800'
                                        }`}>
                                        <p className="font-semibold">
                                            {finalResult.predicted_transmitter === parseInt(currentSignal.signalId)
                                                ? 'CORRECT PREDICTION!'
                                                : 'Incorrect prediction'
                                            }
                                        </p>
                                    </div>
                                )}

                                <p className="text-sm text-gray-500">
                                    Processing time: {finalResult.processing_time_ms.toFixed(1)}ms
                                </p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Panel: Live CNN Visualization */}
                <div className="lg:col-span-2">
                    <div className="bg-gray-50 p-6 rounded-lg border">
                        <div className="flex items-center gap-2 mb-4">
                            <Layers className="text-purple-600" size={24} />
                            <h3 className="font-semibold text-lg">Live CNN Processing</h3>
                        </div>

                        {layerData.length === 0 && !isVisualizing && (
                            <div className="text-center py-12 text-gray-500">
                                <Layers size={48} className="mx-auto mb-4 opacity-30" />
                                <p>Select a signal and click "Classify with Live CNN View"</p>
                                <p className="text-sm">to see how our model processes RF signals.</p>
                            </div>
                        )}

                        {(isVisualizing || layerData.length > 0) && (
                            <>
                                {/* Progress Bar */}
                                <div className="mb-6">
                                    <div className="flex justify-between text-sm text-gray-600 mb-2">
                                        {["Input", "Conv1", "Conv2", "Conv3", "Conv4", "Output"].map((name, idx) => (
                                            <div
                                                key={idx}
                                                className={`text-center transition-colors ${idx <= currentLayer ? 'text-purple-600 font-semibold' : ''
                                                    }`}
                                            >
                                                {name}
                                            </div>
                                        ))}
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-3">
                                        <div
                                            className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full transition-all duration-700 ease-out"
                                            style={{ width: `${((currentLayer + 1) / 6) * 100}%` }}
                                        ></div>
                                    </div>
                                </div>

                                {/* Layer Visualizations */}
                                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                                    {layerData.map((layer, index) => renderLayerVisualization(layer, index))}
                                </div>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Instructions */}
            <div className="mt-8 text-center text-gray-600 text-sm">
                <p></p>
            </div>
        </div>
    );
};

export default SignalShuffler;