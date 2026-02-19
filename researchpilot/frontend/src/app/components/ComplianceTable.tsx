import { ChevronDown, ChevronRight } from 'lucide-react';
import { useState } from 'react';
import React from 'react';

const complianceData = [
  {
    id: 1,
    category: 'Cloud Infrastructure',
    status: 'Compliant',
    lastAudit: '2026-02-15',
    score: 98,
    details: 'All cloud resources properly tagged and monitored. Auto-scaling configured correctly.',
  },
  {
    id: 2,
    category: 'Vendor Management',
    status: 'At Risk',
    lastAudit: '2026-02-10',
    score: 72,
    details: '3 vendors missing updated certifications. Contract renewals pending review.',
  },
  {
    id: 3,
    category: 'Data Retention',
    status: 'Compliant',
    lastAudit: '2026-02-12',
    score: 95,
    details: 'Backup policies aligned with regulatory requirements. Encryption standards met.',
  },
  {
    id: 4,
    category: 'Procurement Process',
    status: 'Non-Compliant',
    lastAudit: '2026-02-08',
    score: 58,
    details: 'Multiple purchase orders missing required approvals. Urgency threshold exceeded.',
  },
  {
    id: 5,
    category: 'Cost Allocation',
    status: 'Compliant',
    lastAudit: '2026-02-14',
    score: 92,
    details: 'Department budgets properly tracked. Chargeback system functioning as expected.',
  },
];

export function ComplianceTable() {
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  const getStatusStyle = (status: string) => {
    switch (status) {
      case 'Compliant':
        return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      case 'At Risk':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'Non-Compliant':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      default:
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-emerald-400';
    if (score >= 70) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-500/10 to-violet-600/10 p-[1px]">
      <div className="h-full rounded-2xl bg-[#0B1220]/90 backdrop-blur-xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg text-white">Compliance Status</h3>
            <p className="text-sm text-white/50 mt-1">Real-time monitoring across all categories</p>
          </div>
          <div className="text-sm text-white/60">
            <span className="text-emerald-400">3</span> Compliant • 
            <span className="text-orange-400 ml-1">1</span> At Risk • 
            <span className="text-red-400 ml-1">1</span> Non-Compliant
          </div>
        </div>

        <div className="overflow-hidden rounded-lg border border-white/10">
          <table className="w-full">
            <thead className="bg-white/5">
              <tr>
                <th className="text-left px-4 py-3 text-xs text-white/50 uppercase tracking-wider">Category</th>
                <th className="text-left px-4 py-3 text-xs text-white/50 uppercase tracking-wider">Status</th>
                <th className="text-left px-4 py-3 text-xs text-white/50 uppercase tracking-wider">Last Audit</th>
                <th className="text-left px-4 py-3 text-xs text-white/50 uppercase tracking-wider">Score</th>
                <th className="w-12"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {complianceData.flatMap((item) => {
                const rows = [
                  <tr
                    key={`main-${item.id}`}
                    className="hover:bg-white/5 cursor-pointer transition-colors"
                    onClick={() => setExpandedRow(expandedRow === item.id ? null : item.id)}
                  >
                    <td className="px-4 py-4 text-sm text-white">{item.category}</td>
                    <td className="px-4 py-4">
                      <span className={`px-3 py-1 rounded-full text-xs border ${getStatusStyle(item.status)}`}>
                        {item.status}
                      </span>
                    </td>
                    <td className="px-4 py-4 text-sm text-white/60">{item.lastAudit}</td>
                    <td className="px-4 py-4">
                      <span className={`text-sm ${getScoreColor(item.score)}`}>{item.score}%</span>
                    </td>
                    <td className="px-4 py-4">
                      {expandedRow === item.id ? (
                        <ChevronDown className="w-4 h-4 text-white/40" />
                      ) : (
                        <ChevronRight className="w-4 h-4 text-white/40" />
                      )}
                    </td>
                  </tr>
                ];
                
                if (expandedRow === item.id) {
                  rows.push(
                    <tr key={`detail-${item.id}`}>
                      <td colSpan={5} className="px-4 py-4 bg-white/5">
                        <div className="text-sm text-white/70 p-4 bg-white/5 rounded-lg border border-white/10">
                          <span className="text-white/50">Details: </span>
                          {item.details}
                        </div>
                      </td>
                    </tr>
                  );
                }
                
                return rows;
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}