import { TrendingUp, TrendingDown } from 'lucide-react';

interface KPICardProps {
  label: string;
  value: string;
  trend: number;
  gradientFrom: string;
  gradientTo: string;
}

export function KPICard({ label, value, trend, gradientFrom, gradientTo }: KPICardProps) {
  const isPositive = trend > 0;

  return (
    <div className={`relative overflow-hidden rounded-2xl bg-gradient-to-br ${gradientFrom} ${gradientTo} p-[1px]`}>
      <div className="h-full rounded-2xl bg-[#0B1220]/90 backdrop-blur-xl p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-white/50 mb-2">{label}</p>
            <p className="text-3xl text-white mb-3">{value}</p>
            <div className={`flex items-center gap-1 ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
              {isPositive ? (
                <TrendingUp className="w-4 h-4" />
              ) : (
                <TrendingDown className="w-4 h-4" />
              )}
              <span className="text-sm">{Math.abs(trend)}%</span>
              <span className="text-xs text-white/40 ml-1">vs last month</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
