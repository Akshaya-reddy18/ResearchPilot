import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

const data = [
  { month: 'Jan', actual: 24.5, forecast: 24.5, anomaly: false },
  { month: 'Feb', actual: 26.2, forecast: 25.8, anomaly: false },
  { month: 'Mar', actual: 25.8, forecast: 27.1, anomaly: false },
  { month: 'Apr', actual: 32.1, forecast: 28.4, anomaly: true },
  { month: 'May', actual: 28.9, forecast: 29.7, anomaly: false },
  { month: 'Jun', actual: 31.2, forecast: 31.0, anomaly: false },
  { month: 'Jul', actual: 33.5, forecast: 32.3, anomaly: false },
  { month: 'Aug', actual: 41.8, forecast: 33.6, anomaly: true },
  { month: 'Sep', actual: 35.2, forecast: 34.9, anomaly: false },
  { month: 'Oct', actual: null, forecast: 36.2, anomaly: false },
  { month: 'Nov', actual: null, forecast: 37.5, anomaly: false },
  { month: 'Dec', actual: null, forecast: 38.8, anomaly: false },
];

export function CostTrendChart() {
  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-500/10 to-violet-600/10 p-[1px]">
      <div className="h-full rounded-2xl bg-[#0B1220]/90 backdrop-blur-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg text-white">Cost Trend Analysis</h3>
            <p className="text-sm text-white/50 mt-1">Monthly operational costs with AI forecast</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span className="text-xs text-white/60">Actual</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-violet-500"></div>
              <span className="text-xs text-white/60">Forecast</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]"></div>
              <span className="text-xs text-white/60">Anomaly</span>
            </div>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="actualGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis 
              dataKey="month" 
              stroke="rgba(255,255,255,0.3)" 
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
            />
            <YAxis 
              stroke="rgba(255,255,255,0.3)" 
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12 }}
              tickFormatter={(value) => `₹${value}L`}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(11, 18, 32, 0.95)', 
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                color: '#fff'
              }}
              formatter={(value: number) => [`₹${value}L`, '']}
            />
            <Area 
              type="monotone" 
              dataKey="actual" 
              stroke="#3B82F6" 
              strokeWidth={3}
              fill="url(#actualGradient)"
              dot={(props: any) => {
                const { cx, cy, payload, index } = props;
                if (payload.anomaly) {
                  return (
                    <g key={`anomaly-${index}`}>
                      <circle key={`outer-${index}`} cx={cx} cy={cy} r={8} fill="#EF4444" opacity={0.2} />
                      <circle key={`inner-${index}`} cx={cx} cy={cy} r={4} fill="#EF4444" stroke="#fff" strokeWidth={2} />
                    </g>
                  );
                }
                return <circle key={`dot-${index}`} cx={cx} cy={cy} r={4} fill="#3B82F6" stroke="#fff" strokeWidth={2} />;
              }}
            />
            <Line 
              type="monotone" 
              dataKey="forecast" 
              stroke="#8B5CF6" 
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}