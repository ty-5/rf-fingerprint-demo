import React from 'react';

const WaveformChart = ({ data, title, color, height = 80, width = 200 }) => {
    if (!data || !Array.isArray(data)) return null;

    const padding = 10;
    const innerWidth = width - padding * 2;
    const innerHeight = height - padding * 2;

    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;

    const points = data.slice(0, Math.min(data.length, 64)).map((val, idx) => {
        const x = padding + (idx / (Math.min(data.length, 64) - 1)) * innerWidth;
        const y = padding + ((maxVal - val) / range) * innerHeight;
        return `${x},${y}`;
    }).join(' ');

    return (
        <div className="text-center">
            <div className="text-xs font-medium text-gray-700 mb-1">{title}</div>
            <svg width={width} height={height} className="border bg-white rounded">
                <polyline
                    points={points}
                    fill="none"
                    stroke={color}
                    strokeWidth="1.5"
                />
            </svg>
        </div>
    );
};

export default WaveformChart;