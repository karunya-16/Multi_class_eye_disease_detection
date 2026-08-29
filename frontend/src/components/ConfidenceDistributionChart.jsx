import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function ConfidenceDistributionChart({ data }) {
  return (
    <div
      className="h-72 min-w-0 w-full"
      role="img"
      aria-label="Model Confidence distribution from stored MySQL analyses"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d5e1de" />
          <XAxis dataKey="bucket" tick={{ fill: '#5b6f6b', fontSize: 12 }} />
          <YAxis allowDecimals={false} tick={{ fill: '#5b6f6b', fontSize: 12 }} />
          <Tooltip
            formatter={(value) => [value, 'Analyses']}
            labelFormatter={(label) => `Model Confidence ${label}`}
            contentStyle={{ borderRadius: 12, borderColor: '#d5e1de' }}
          />
          <Bar dataKey="count" fill="#3d8b80" radius={[6, 6, 0, 0]} maxBarSize={56} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
