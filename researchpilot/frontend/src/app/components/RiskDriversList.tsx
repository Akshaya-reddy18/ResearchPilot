import { AlertTriangle } from 'lucide-react';

const riskDrivers = [
  { rank: 1, driver: 'Cloud Infrastructure Overprovisioning', severity: 'Critical', impact: '₹4.2L/month' },
  { rank: 2, driver: 'Unused Software Licenses', severity: 'High', impact: '₹2.8L/month' },
  { rank: 3, driver: 'Non-compliant Procurement Process', severity: 'High', impact: '₹1.9L/month' },
  { rank: 4, driver: 'Manual Workflow Inefficiencies', severity: 'Medium', impact: '₹1.2L/month' },
  { rank: 5, driver: 'Data Storage Redundancy', severity: 'Medium', impact: '₹0.8L/month' },
];

export function RiskDriversList() {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'Critical':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'High':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'Medium':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      default:
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    }
  };

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-500/10 to-violet-600/10 p-[1px]">
      <div className="h-full rounded-2xl bg-[#0B1220]/90 backdrop-blur-xl p-6">
        <div className="flex items-center gap-3 mb-6">
          <AlertTriangle className="w-5 h-5 text-orange-400" />
          <h3 className="text-lg text-white">Top 5 Risk Drivers</h3>
        </div>

        <div className="space-y-4">
          {riskDrivers.map((item) => (
            <div
              key={item.rank}
              className="flex items-start gap-4 p-4 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 transition-all cursor-pointer"
            >
              <div className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 text-white text-sm shrink-0">
                {item.rank}
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-sm text-white mb-2">{item.driver}</h4>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-1 rounded text-xs border ${getSeverityColor(item.severity)}`}>
                    {item.severity}
                  </span>
                  <span className="text-xs text-white/50">•</span>
                  <span className="text-xs text-white/60">{item.impact}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
