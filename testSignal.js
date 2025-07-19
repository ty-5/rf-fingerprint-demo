// Load and examine the signal data
import fs from 'fs';

const data = JSON.parse(fs.readFileSync('rf_signals.json', 'utf8'));

// Look at first signal
const firstSignal = data["0"];
console.log("First signal shape:", firstSignal.length, "x", firstSignal[0].length);
console.log("I channel first 5 values:", firstSignal[0].slice(0, 5));
console.log("Q channel first 5 values:", firstSignal[1].slice(0, 5));
console.log("Total signals available:", Object.keys(data).length);