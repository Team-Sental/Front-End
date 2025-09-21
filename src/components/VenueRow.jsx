import React from "react";
import { getBarColor } from "../utils/color";

export function VenueRow({ venues }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(6, minmax(0, 1fr))",
        gap: 12,
      }}
    >
      {venues.map((v) => {
        const pct = v.capacity
          ? Math.min(100, Math.round((v.current / v.capacity) * 100))
          : 0;
        const radius = 38; // circle radius
        const circumference = 2 * Math.PI * radius;
        const strokeDashoffset = circumference - (pct / 100) * circumference;
        return (
          <div
            key={v.name}
            style={{
              background: "#1f2937",
              padding: 12,
              borderRadius: 10,
              color: "white",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <div
              style={{
                fontWeight: 700,
                marginBottom: 6,
                textTransform: "capitalize",
              }}
            >
              {v.name.replace(/_/g, " ")}
            </div>
            <div style={{ fontSize: 14, color: "#cbd5e1", marginBottom: 8 }}>
              {v.current.toLocaleString()} / {v.capacity.toLocaleString()}
            </div>
            <div
              style={{
                position: "relative",
                width: 100,
                height: 100,
                marginTop: 4,
              }}
            >
              <svg
                width={100}
                height={100}
                style={{ transform: "rotate(-90deg)" }}
              >
                <circle
                  cx={50}
                  cy={50}
                  r={radius}
                  stroke="#374151"
                  strokeWidth={10}
                  fill="none"
                />
                <circle
                  cx={50}
                  cy={50}
                  r={radius}
                  stroke={getBarColor(pct)}
                  strokeWidth={10}
                  fill="none"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  style={{ transition: "stroke-dashoffset 0.6s ease" }}
                />
              </svg>
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexDirection: "column",
                  // Text shown upright (SVG rotated -90deg, so omit counter-rotation for readability)
                  fontSize: 14,
                  fontWeight: 600,
                  color: getBarColor(pct),
                }}
              >
                <span>{pct}%</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default VenueRow;
