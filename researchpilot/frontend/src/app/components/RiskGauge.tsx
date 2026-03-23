import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

export function RiskGauge() {
  const riskScore = 67;
  const data = [
    { name: 'Risk', value: riskScore },
    { name: 'Remaining', value: 100 - riskScore }
  ];

  const COLORS = ['#8B5CF6', 'rgba(255,255,255,0.05)'];

  const getRiskLevel = (score: number) => {
    if (score >= 80) return { level: 'Critical', color: 'text-red-400' };
    if (score >= 60) return { level: 'High', color: 'text-orange-400' };
    if (score >= 40) return { level: 'Medium', color: 'text-yellow-400' };
    return { level: 'Low', color: 'text-emerald-400' };
  };

  const risk = getRiskLevel(riskScore);

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-violet-500/10 to-blue-500/10 p-[1px]">
      <div className="h-full rounded-2xl bg-[#0B1220]/90 backdrop-blur-xl p-6">
        <h3 className="text-lg text-white mb-2">Risk Score</h3>
        <p className="text-sm text-white/50 mb-6">Enterprise risk assessment</p>

        <div className="relative">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="100%"
                startAngle={180}
                endAngle={0}
                innerRadius={80}
                outerRadius={100}
                dataKey="value"
                stroke="none"
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center">
            <div className="text-5xl text-white mb-1">{riskScore}</div>
            <div className={`text-sm ${risk.color}`}>{risk.level} Risk</div>
          </div>
        </div>

        <div className="mt-8 space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-white/50">Low</span>
            <span className="text-xs text-white/50">Medium</span>
            <span className="text-xs text-white/50">High</span>
            <span className="text-xs text-white/50">Critical</span>
          </div>
          <div className="h-2 rounded-full bg-gradient-to-r from-emerald-500 via-yellow-500 via-orange-500 to-red-500 opacity-30"></div>
        </div>
      </div>
    </div>
  );
}
