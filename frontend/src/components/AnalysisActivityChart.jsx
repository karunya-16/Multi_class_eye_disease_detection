import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function AnalysisActivityChart({ data, granularity = 'day' }) {
  return (
    <div
      className="h-72 min-w-0 w-full"
      role="img"
      aria-label="Analysis activity over time from stored MySQL timestamps"
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d5e1de" />
          <XAxis
            dataKey="bucket"
            tick={{ fill: '#5b6f6b', fontSize: 11 }}
            interval="preserveStartEnd"
          />
          <YAxis allowDecimals={false} tick={{ fill: '#5b6f6b', fontSize: 12 }} />
          <Tooltip
            formatter={(value) => [value, 'Analyses']}
            labelFormatter={(label) => (granularity === 'hour' ? `Hour ${label}` : `Date ${label}`)}
            contentStyle={{ borderRadius: 12, borderColor: '#d5e1de' }}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke="#0f6a5c"
            fill="#d9efe9"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
