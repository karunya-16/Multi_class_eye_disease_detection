import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { CLASS_NAMES } from '@/lib/constants';
import { probabilityToPercent } from '@/lib/utils';

export default function ProbabilityChart({ probabilities, predictedLabel }) {
  const data = CLASS_NAMES.map((name) => ({
    name,
    probability: Number(probabilities?.[name]),
    percent: Number(probabilityToPercent(probabilities?.[name]).toFixed(2)),
    predicted: name === predictedLabel,
  }));

  return (
    <div
      className="h-72 min-w-0 w-full max-w-full"
      role="img"
      aria-label="Four-class probability bar chart from the FastAPI response"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#d5e1de" />
          <XAxis dataKey="name" tick={{ fill: '#5b6f6b', fontSize: 12 }} />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
            tick={{ fill: '#5b6f6b', fontSize: 12 }}
          />
          <Tooltip
            formatter={(value, _name, item) => {
              const raw = item?.payload?.probability;
              return [`${value}% (value ${raw})`, 'class_probabilities'];
            }}
            contentStyle={{ borderRadius: 12, borderColor: '#d5e1de' }}
          />
          <Bar dataKey="percent" radius={[6, 6, 0, 0]} maxBarSize={48}>
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={entry.predicted ? '#0f6a5c' : '#9bb9b3'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
