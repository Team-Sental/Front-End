import React from "react";
import { getBarColor } from "./getBarColor";

function AttendanceCard({ attendance, capacity }) {
  const percent = capacity ? Math.min(100, Math.round((attendance / capacity) * 100)) : 0;
  const color = getBarColor(percent);

  return (
    <div style={{ background: "#111827", padding: 16, borderRadius: 10, color: "white" }}>
      <div style={{ fontSize: 14, color: "#cbd5e1" }}>Live Attendance</div>
      <div style={{ fontSize: 28, fontWeight: 700 }}>{attendance.toLocaleString()}</div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>
        of {capacity.toLocaleString()} total capacity
      </div>
      <div style={{ width: "100%", background: "#374151", height: 12, borderRadius: 8 }}>
        <div style={{ width: `${percent}%`, height: "100%", borderRadius: 8, background: color }} />
      </div>
      <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 6 }}>{percent}%</div>
    </div>
  );
}

export default AttendanceCard;