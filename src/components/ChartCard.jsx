import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";

export function ChartCard({ graphData }) {
  return (
    <div
      style={{
        background: "#111827",
        padding: 16,
        borderRadius: 10,
        color: "white",
      }}
    >
      <div style={{ fontSize: 14, color: "#cbd5e1", marginBottom: 6 }}>
        Attendance Trend
      </div>
      <div style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={graphData}>
            <CartesianGrid stroke="#2d3748" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual Data"
              stroke="#4A90E2"
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicted Data"
              stroke="#F59E0B"
              strokeDasharray="5 5"
              dot={{ r: 4 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default ChartCard;
