import React, { useState, useEffect } from 'react';
import { Shuffle, Play, Eye, EyeOff } from 'lucide-react';

const SignalShuffler = () => {
    const [signalData, setSignalData] = useState(null);
    const [currentSignal, setCurrentSignal] = useState(null);
    const [isClassifying, setIsClassifying] = useState(false);
    const [result, setResult] = useState(null);
    const [showSignalId, setShowSignalId] = useState(false);

    // Load the JSON signal data on component mount
    useEffect(() => {
        const loadSignalData = async () => {
            try {
                console.log('Loading signal data...');
                const response = await fetch('/rf_signals.json');
                if (!response.ok) throw new Error('Failed to load signal data');

                const data = await response.json();
                setSignalData(data);
                console.log(`Loaded ${Object.keys(data).length} transmitters`);

                // DEBUG: Show the structure
                const firstKey = Object.keys(data)[0];
                console.log(`First transmitter (${firstKey}) has ${data[firstKey].length} samples`);
                if (data[firstKey].length > 0) {
                    console.log(`First sample shape:`, data[firstKey][0].length, 'x', data[firstKey][0][0]?.length);
                }
            } catch (error) {
                console.error('Failed to load signal data:', error);
            }
        };

        loadSignalData();
    }, []);

    // Get a random signal from the collection
    const getRandomSignal = () => {
        if (!signalData) return;

        const transmitterKeys = Object.keys(signalData);
        const randomTransmitterKey = transmitterKeys[Math.floor(Math.random() * transmitterKeys.length)];

        // Get random sample from that transmitter
        const transmitterSamples = signalData[randomTransmitterKey];
        const randomSampleIndex = Math.floor(Math.random() * transmitterSamples.length);
        const selectedSample = transmitterSamples[randomSampleIndex];

        console.log(`Selected transmitter ${randomTransmitterKey}, sample ${randomSampleIndex}`);
        console.log(`Sample shape:`, selectedSample.length, 'x', selectedSample[0]?.length);
        console.log(`Sample type:`, typeof selectedSample, Array.isArray(selectedSample));
        console.log(`First few I values:`, selectedSample[0]?.slice(0, 5));
        console.log(`First few Q values:`, selectedSample[1]?.slice(0, 5));

        setCurrentSignal({
            signal: selectedSample,  // This is already [I_channel, Q_channel]
            signalId: randomTransmitterKey,
            sampleIndex: randomSampleIndex
        });

        setResult(null);
        setShowSignalId(false);
    };

    // Send signal to FastAPI backend for classification
    const classifySignal = async () => {
        if (!currentSignal) return;

        setIsClassifying(true);
        setResult(null);

        // The signal is already in [I_channel, Q_channel] format!
        // Just need to ensure it's the right length (128 samples)
        let processedSignal = currentSignal.signal;

        console.log('🔍 Original signal shape:', processedSignal.length, 'x', processedSignal[0]?.length);

        // Pad or truncate to 128 samples if needed
        if (processedSignal[0]?.length !== 128) {
            console.log(`🔧 Adjusting signal length from ${processedSignal[0]?.length} to 128`);

            processedSignal = processedSignal.map(channel => {
                if (channel.length < 128) {
                    // Pad with zeros
                    const padding = new Array(128 - channel.length).fill(0);
                    return [...channel, ...padding];
                } else if (channel.length > 128) {
                    // Truncate
                    return channel.slice(0, 128);
                }
                return channel;
            });
        }

        console.log('🔍 Final signal shape:', processedSignal.length, 'x', processedSignal[0]?.length);
        console.log('🔍 I channel sample:', processedSignal[0]?.slice(0, 5));
        console.log('🔍 Q channel sample:', processedSignal[1]?.slice(0, 5));

        const requestData = {
            signal: processedSignal.map(channel =>
                channel.map(val => parseFloat(val)) // Ensure all values are proper floats
            ),
            get_layers: false,
            metadata: {
                true_transmitter: currentSignal.signalId,
                sample_index: currentSignal.sampleIndex
            }
        };

        try {
            console.log('Sending signal to classifier...');

            const response = await fetch('http://localhost:8000/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('FastAPI Error Response:', errorText);

                try {
                    const errorJson = JSON.parse(errorText);
                    console.error('Parsed Error:', errorJson);
                } catch (e) {
                    console.error('Raw Error Text:', errorText);
                }

                throw new Error(`Classification failed: ${response.status}`);
            }

            const result = await response.json();
            setResult(result);
            console.log('✅ Classification complete:', result);

            // Compare prediction vs ground truth
            const predicted = result.predicted_transmitter;
            const actual = parseInt(currentSignal.signalId);
            const isCorrect = predicted === actual;

            console.log(`Prediction: ${predicted}, Actual: ${actual}, Correct: ${isCorrect ? '✅' : '❌'}`);

        } catch (error) {
            console.error('Classification failed:', error);
        } finally {
            setIsClassifying(false);
        }
    };

    // Toggle showing the signal ID (ground truth reveal)
    const toggleSignalId = () => {
        setShowSignalId(!showSignalId);
    };

    return (
        <div className="signal-shuffler p-6 max-w-2xl mx-auto">
            <h1 className="text-3xl font-bold mb-6 text-center">
                RF Signal Classifier Demo
            </h1>

            {/* Signal Data Status */}
            <div className="mb-6 p-4 bg-gray-100 rounded-lg">
                {signalData ? (
                    <p className="text-green-600">
                        Loaded {Object.keys(signalData).length} RF signals ready for classification
                    </p>
                ) : (
                    <p className="text-gray-600">📡 Loading signal database...</p>
                )}
            </div>

            {/* Random Signal Picker */}
            <div className="mb-6 text-center">
                <button
                    onClick={getRandomSignal}
                    disabled={!signalData}
                    className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                    <Shuffle size={20} />
                    Get Random Signal
                </button>
            </div>

            {/* Current Signal Info */}
            {currentSignal && (
                <div className="mb-6 p-4 bg-blue-50 rounded-lg">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="font-semibold">Current Signal:</h3>
                            <p className="text-sm text-gray-600">
                                Shape: {currentSignal.signal.length} channels × {currentSignal.signal[0]?.length} samples
                            </p>
                            {showSignalId && (
                                <div className="text-sm font-mono text-blue-600">
                                    <p>True Transmitter: #{currentSignal.signalId}</p>
                                    <p>Sample Index: {currentSignal.sampleIndex}</p>
                                </div>
                            )}
                        </div>

                        <div className="flex gap-2">
                            <button
                                onClick={classifySignal}
                                disabled={isClassifying}
                                className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition-colors"
                            >
                                <Play size={16} />
                                {isClassifying ? 'Classifying...' : 'Classify Signal'}
                            </button>

                            {result && (
                                <button
                                    onClick={toggleSignalId}
                                    className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
                                >
                                    {showSignalId ? <EyeOff size={16} /> : <Eye size={16} />}
                                    {showSignalId ? 'Hide ID' : 'Reveal Truth'}
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Classification Results */}
            {result && (
                <div className="mb-6 p-4 bg-green-50 rounded-lg">
                    <h3 className="font-semibold mb-2">Classification Results:</h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <p className="text-sm text-gray-600">Predicted Transmitter:</p>
                            <p className="text-2xl font-bold text-green-600">
                                Transmitter {result.predicted_transmitter}
                            </p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-600">Confidence:</p>
                            <p className="text-2xl font-bold text-green-600">
                                {(result.confidence * 100).toFixed(1)}%
                            </p>
                        </div>
                    </div>

                    {/* Show accuracy check if ground truth is revealed */}
                    {showSignalId && (
                        <div className="mt-3 p-3 bg-blue-100 rounded">
                            <p className="text-sm font-semibold">
                                {result.predicted_transmitter === parseInt(currentSignal.signalId) ?
                                    'CORRECT PREDICTION!' :
                                    'Incorrect prediction'
                                }
                            </p>
                        </div>
                    )}

                    <p className="text-sm text-gray-500 mt-2">
                        Processing time: {result.processing_time_ms.toFixed(1)}ms
                    </p>
                </div>
            )}

            {/* Instructions */}
            <div className="text-center text-gray-600 text-sm">
                <p>Click "Get Random Signal" to select an unknown RF signal from the ORACLE dataset,</p>
                <p>then "Classify Signal" to see our 98% accuracy CNN model identify the transmitter!</p>
            </div>
        </div>
    );
};

export default SignalShuffler;