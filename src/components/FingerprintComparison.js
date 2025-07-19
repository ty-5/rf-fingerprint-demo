import React, { useState } from 'react';
import WaveformChart from './WaveformChart';

const FingerprintComparison = ({ signalData }) => {
    const [selectedTransmitter, setSelectedTransmitter] = useState(null);
    const [comparisonSamples, setComparisonSamples] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [comparisonResults, setComparisonResults] = useState([]);

    // Prepare signal for API
    const prepareSignalForComparison = (signal) => {
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

    // Generate fingerprint comparison for selected transmitter
    const generateFingerprints = async (transmitterId) => {
        setIsProcessing(true);
        setSelectedTransmitter(transmitterId);
        setComparisonResults([]);

        try {
            console.log(`Analyzing fingerprints for Transmitter ${transmitterId}`);

            // Select 3 random samples from the transmitter
            const transmitterSamples = signalData[transmitterId.toString()];
            const sampleIndices = [];

            // Get 3 unique random indices
            while (sampleIndices.length < 3) {
                const randomIndex = Math.floor(Math.random() * transmitterSamples.length);
                if (!sampleIndices.includes(randomIndex)) {
                    sampleIndices.push(randomIndex);
                }
            }

            const selectedSamples = sampleIndices.map(idx => ({
                signal: transmitterSamples[idx],
                index: idx
            }));

            setComparisonSamples(selectedSamples);
            console.log(`Selected samples: ${sampleIndices.join(', ')}`);

            // Process each sample through ConvBlock3
            const results = [];
            for (let i = 0; i < selectedSamples.length; i++) {
                const sample = selectedSamples[i];

                console.log(`Processing sample ${sample.index} through ConvBlock3...`);

                // Prepare signal
                const processedSignal = prepareSignalForComparison(sample.signal);

                // Get ConvBlock3 features (layer index 3)
                const response = await fetch('http://localhost:8000/classify/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        signal: processedSignal,
                        layer_index: 3  // ConvBlock3
                    })
                });

                if (response.ok) {
                    const layerResult = await response.json();
                    results.push({
                        sampleIndex: sample.index,
                        rawSignal: sample.signal,
                        features: layerResult.feature_maps,
                        layerInfo: layerResult.layer_info
                    });
                    console.log(`Sample ${sample.index} processed successfully`);
                } else {
                    console.error(`Failed to process sample ${sample.index}`);
                }

                // Add delay for visual effect
                await new Promise(resolve => setTimeout(resolve, 1000));
            }

            setComparisonResults(results);
            console.log(`Fingerprint analysis complete for Transmitter ${transmitterId}`);

        } catch (error) {
            console.error('Fingerprint comparison failed:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    // Render raw signal waveform
    const renderRawSignal = (signal, label) => {
        if (!signal || !Array.isArray(signal)) return null;

        return (
            <div className="text-center">
                <h4 className="font-semibold text-sm mb-3">{label}</h4>
                <div className="space-y-3">
                    <WaveformChart
                        data={signal[0]}
                        title="I Channel"
                        color="#EF4444"
                        height={70}
                        width={220}
                    />
                    <WaveformChart
                        data={signal[1]}
                        title="Q Channel"
                        color="#10B981"
                        height={70}
                        width={220}
                    />
                </div>
            </div>
        );
    };

    // Render ConvBlock3 features
    const renderFingerprint = (features, label) => {
        if (!features || !Array.isArray(features)) return null;

        return (
            <div className="text-center">
                <h4 className="font-semibold text-sm mb-3">{label}</h4>
                <div className="grid grid-cols-2 gap-2">
                    {features.slice(0, 4).map((feature, idx) => (
                        <WaveformChart
                            key={idx}
                            data={feature}
                            title={`Feature ${idx + 1}`}
                            color="#8B5CF6"
                            height={50}
                            width={100}
                        />
                    ))}
                </div>
                <div className="text-xs text-gray-500 mt-2">
                    4 of {features.length} features shown
                </div>
            </div>
        );
    };

    return (
        <div className="fingerprint-comparison bg-yellow-50 p-6 rounded-lg border border-yellow-200">
            {/* Header */}
            <div className="flex items-center gap-2 mb-4">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <h2 className="font-bold text-xl">🔍 Signal Fingerprint Analysis</h2>
            </div>

            <div className="bg-yellow-100 p-4 rounded mb-6">
                <p className="text-sm text-yellow-800">
                    <strong>Demonstration:</strong> Select any transmitter to see how our CNN extracts consistent
                    "fingerprints" from different signal captures. Raw signals may look completely different,
                    but ConvBlock3 features reveal the hidden patterns that identify each transmitter.
                </p>
            </div>

            {/* Transmitter Selection */}
            <div className="mb-6">
                <h3 className="font-semibold mb-3">Select Transmitter to Analyze:</h3>
                <div className="grid grid-cols-8 gap-2">
                    {Array.from({ length: 16 }, (_, i) => (
                        <button
                            key={i}
                            onClick={() => generateFingerprints(i)}
                            disabled={isProcessing || !signalData}
                            className={`px-3 py-2 text-sm font-medium rounded border transition-all duration-200 ${selectedTransmitter === i
                                    ? 'bg-yellow-500 text-white border-yellow-600 shadow-md'
                                    : 'bg-white text-gray-700 border-gray-300 hover:bg-yellow-100 hover:border-yellow-400'
                                } disabled:bg-gray-200 disabled:cursor-not-allowed disabled:text-gray-400`}
                        >
                            T{i}
                        </button>
                    ))}
                </div>
            </div>

            {/* Processing Status */}
            {isProcessing && (
                <div className="text-center py-6">
                    <div className="inline-flex items-center gap-3 text-yellow-700">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-600"></div>
                        <span className="font-medium">
                            Analyzing Transmitter {selectedTransmitter} fingerprints...
                        </span>
                    </div>
                    <p className="text-sm text-yellow-600 mt-2">
                        Processing {comparisonResults.length + 1} of 3 samples through ConvBlock3
                    </p>
                </div>
            )}

            {/* Results Display */}
            {comparisonResults.length > 0 && !isProcessing && (
                <div className="space-y-8">
                    {/* Header */}
                    <div className="text-center">
                        <h3 className="font-bold text-xl mb-2">
                            Transmitter {selectedTransmitter} Analysis Results
                        </h3>
                        <p className="text-gray-600">
                            Three random signal captures from samples: {comparisonResults.map(r => `#${r.sampleIndex}`).join(', ')}
                        </p>
                    </div>

                    {/* Raw Signals Comparison */}
                    <div className="bg-white p-6 rounded-lg border">
                        <h4 className="font-semibold text-lg mb-4 text-center flex items-center justify-center gap-2">
                            📡 Raw Signal Captures <span className="text-sm text-gray-500">(What Humans See)</span>
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {comparisonResults.map((result, idx) => (
                                <div key={idx} className="p-4 bg-gray-50 rounded border">
                                    {renderRawSignal(
                                        result.rawSignal,
                                        `Sample ${result.sampleIndex}`
                                    )}
                                </div>
                            ))}
                        </div>
                        <div className="text-center mt-4">
                            <p className="text-sm text-gray-600 bg-gray-100 inline-block px-3 py-1 rounded">
                                These signals look completely different to human observers
                            </p>
                        </div>
                    </div>

                    {/* ConvBlock3 Features Comparison */}
                    <div className="bg-purple-50 p-6 rounded-lg border border-purple-200">
                        <h4 className="font-semibold text-lg mb-4 text-center flex items-center justify-center gap-2">
                            ConvBlock3 Features <span className="text-sm text-purple-600">(What the AI Sees)</span>
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {comparisonResults.map((result, idx) => (
                                <div key={idx} className="p-4 bg-white rounded border border-purple-200">
                                    {renderFingerprint(
                                        result.features,
                                        `Sample ${result.sampleIndex}`
                                    )}
                                </div>
                            ))}
                        </div>
                        <div className="text-center mt-4">
                            <p className="text-sm text-purple-700 bg-purple-100 inline-block px-3 py-1 rounded">
                                The AI extracts similar patterns - this is the "RF fingerprint"!
                            </p>
                        </div>
                    </div>

                    {/* Insight Box */}
                    <div className="bg-green-50 p-6 rounded-lg border border-green-200">
                        <div className="flex items-start gap-3">
                            <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-white text-sm font-bold mt-0.5">
                                ✓
                            </div>
                            <div>
                                <h5 className="font-bold text-green-800 mb-2">Key Discovery</h5>
                                <p className="text-sm text-green-700 leading-relaxed">
                                    This demonstrates the core of RF fingerprinting: our CNN learns to extract
                                    consistent transmitter-specific features from varying signal conditions.
                                    While raw signals differ due to distance, noise, or environmental factors,
                                    the underlying hardware "fingerprint" remains detectable and reliable.
                                </p>
                                <p className="text-sm text-green-700 mt-2 font-medium">
                                    This is why we achieve 98%+ accuracy across different distances and conditions.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Instructions */}
            {comparisonResults.length === 0 && !isProcessing && (
                <div className="text-center py-12 text-gray-500">
                    <div className="text-6xl mb-4">🔍</div>
                    <h4 className="font-semibold text-lg mb-2">Ready to Reveal the Magic?</h4>
                    <p className="text-sm">Select any transmitter above to see how our AI extracts</p>
                    <p className="text-sm">consistent fingerprints from different signal captures</p>
                </div>
            )}
        </div>
    );
};

export default FingerprintComparison;