import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function ClassDistributionChart({ data }) {
  return (
    <div
      className="h-72 min-w-0 w-full"
      role="img"
      aria-label="Predicted class counts from stored MySQL analyses"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d5e1de" />
          <XAxis dataKey="predicted_label" tick={{ fill: '#5b6f6b', fontSize: 12 }} />
          <YAxis allowDecimals={false} tick={{ fill: '#5b6f6b', fontSize: 12 }} />
          <Tooltip
            formatter={(value) => [value, 'Analyses']}
            contentStyle={{ borderRadius: 12, borderColor: '#d5e1de' }}
          />
          <Bar dataKey="count" fill="#0f6a5c" radius={[6, 6, 0, 0]} maxBarSize={48} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
