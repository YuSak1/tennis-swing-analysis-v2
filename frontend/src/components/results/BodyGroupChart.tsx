import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from "recharts";
import type { PlayerSimilarity } from "../../types";

interface BodyGroupChartProps {
  similarities: PlayerSimilarity[];
}

const PLAYER_COLOR_MAP: Record<string, string> = {
  Federer: "#e11d48",
  Nadal: "#f59e0b",
  Djokovic: "#3b82f6",
  Murray: "#8b5cf6",
};

const GROUP_SHORT_NAMES: Record<string, string> = {
  "Racket Arm": "Racket Arm",
  "Non-Racket Arm": "Off Arm",
  "Torso & Rotation": "Torso",
  "Lower Body": "Lower Body",
};

export default function BodyGroupChart({ similarities }: BodyGroupChartProps) {
  if (similarities.length === 0) return null;

  const firstSimilarity = similarities[0];
  if (!firstSimilarity) return null;

  const bodyGroups = Object.keys(firstSimilarity.body_groups);
  const data = bodyGroups.map((group) => {
    const entry: Record<string, string | number> = {
      group: GROUP_SHORT_NAMES[group] ?? group,
    };
    similarities.forEach((s) => {
      entry[s.player] = s.body_groups[group] ?? 0;
    });
    return entry;
  });

  return (
    <div className="rounded-2xl bg-surface-900 border border-surface-800 p-6">
      <h3 className="font-display text-2xl tracking-wide text-white mb-2">
        Body Group Breakdown
      </h3>
      <p className="text-surface-500 text-sm mb-6">
        How each part of your body compares to each player
      </p>

      <ResponsiveContainer width="100%" height={350}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="72%">
          <PolarGrid stroke="#3f3f46" />
          <PolarAngleAxis
            dataKey="group"
            tick={{ fill: "#a1a1aa", fontSize: 13 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: "#52525b", fontSize: 10 }}
            tickCount={4}
          />
          {similarities.map((s) => (
            <Radar
              key={s.player}
              name={s.player}
              dataKey={s.player}
              stroke={PLAYER_COLOR_MAP[s.player] ?? "#10b981"}
              fill={PLAYER_COLOR_MAP[s.player] ?? "#10b981"}
              fillOpacity={0.08}
              strokeWidth={2}
            />
          ))}
          <Tooltip
            contentStyle={{
              background: "#27272a",
              border: "1px solid #3f3f46",
              borderRadius: "12px",
              fontSize: "13px",
            }}
            itemStyle={{ color: "#e4e4e7" }}
            formatter={(value: number) => `${value.toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: "13px", color: "#a1a1aa" }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
