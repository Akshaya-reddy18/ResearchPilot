import { Sparkles, ChevronRight } from 'lucide-react';

const recommendations = [
  {
    title: 'Reduce idle cloud instances to save ₹2.4L annually',
    description: 'Detected 12 underutilized EC2 instances running at <15% capacity',
  },
  {
    title: 'Compliance deviation in procurement workflow',
    description: '3 vendors not meeting regulatory standards - immediate review required',
  },
  {
    title: 'Operational spike detected in Q3',
    description: 'Marketing cloud costs up 47% - suggest budget reallocation',
  },
];

export function AIRecommendations() {
  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-violet-500/10 to-blue-500/10 p-[1px]">
      <div className="h-full rounded-2xl bg-[#0B1220]/90 backdrop-blur-xl p-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg text-white">AI Recommendations</h3>
            <p className="text-xs text-white/50">Actionable insights powered by machine learning</p>
          </div>
        </div>

        <div className="space-y-3">
          {recommendations.map((rec, index) => (
            <div
              key={index}
              className="group p-4 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 hover:border-violet-500/30 transition-all cursor-pointer"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <h4 className="text-sm text-white mb-2 group-hover:text-violet-400 transition-colors">
                    {rec.title}
                  </h4>
                  <p className="text-xs text-white/50">{rec.description}</p>
                </div>
                <ChevronRight className="w-4 h-4 text-white/40 group-hover:text-violet-400 group-hover:translate-x-1 transition-all shrink-0 mt-1" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
