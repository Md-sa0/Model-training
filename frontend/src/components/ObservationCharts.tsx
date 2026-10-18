import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Assessment, FeatureKey, Observation } from "../types/api";
import { featureKeys } from "../types/api";
import { featureLabels, formatDate } from "../lib/format";
import { Card, EmptyState, SelectInput } from "./ui";

const chartColors: Record<FeatureKey, string> = {
  HR: "#65d7b6",
  O2Sat: "#71b6ff",
  Temp: "#e0ad69",
  SBP: "#b59cff",
  MAP: "#e181a1",
  DBP: "#83c2a8",
  Resp: "#d2c476",
  WBC: "#7cb9c8",
  Lactate: "#e29b70",
  Creatinine: "#9cb3ea",
  Platelets: "#d49fd8",
  Age: "#a9bd72",
  Gender: "#cca87c",
  ICULOS: "#76c7c7",
};

export function ObservationCharts({
  observations,
  assessments,
}: {
  observations: Observation[];
  assessments: Assessment[];
}) {
  const [selectedFeature, setSelectedFeature] = useState<FeatureKey>("HR");
  const joined = useMemo(() => {
    const byObservation = new Map(assessments.map((item) => [item.observation_id, item]));
    return [...observations]
      .sort((a, b) => Date.parse(a.observed_at) - Date.parse(b.observed_at))
      .map((observation) => ({
        time: observation.observed_at,
        label: formatDate(observation.observed_at),
        value: observation.values[selectedFeature] ?? null,
        risk: byObservation.get(observation.observation_id)?.risk_score ?? null,
      }));
  }, [observations, assessments, selectedFeature]);

  return (
    <div className="charts-grid">
      <Card
        title="Sinal ao longo do tempo"
        description="Somente observações da página carregada; pontos ausentes não são interpolados."
        action={
          <SelectInput
            className="chart-select"
            aria-label="Selecionar sinal para o gráfico"
            value={selectedFeature}
            onChange={(event) => setSelectedFeature(event.target.value as FeatureKey)}
          >
            {featureKeys.map((feature) => <option key={feature} value={feature}>{feature} · {featureLabels[feature]}</option>)}
          </SelectInput>
        }
      >
        {joined.length < 2 || joined.every((item) => item.value === null) ? (
          <EmptyState title="Dados insuficientes para exibir a série">São necessárias observações com valores disponíveis.</EmptyState>
        ) : (
          <div className="chart-area">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={joined} margin={{ top: 14, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#29343f" strokeDasharray="3 5" vertical={false} />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "#91a0ad", fontSize: 11 }}
                  axisLine={{ stroke: "#34414d" }}
                  tickLine={false}
                  minTickGap={24}
                />
                <YAxis tick={{ fill: "#91a0ad", fontSize: 11 }} axisLine={false} tickLine={false} width={42} />
                <Tooltip
                  contentStyle={{ background: "#111820", border: "1px solid #34414d", borderRadius: 10, color: "#eef3f6" }}
                  formatter={(value) => [value == null ? "Ausente" : String(value), featureLabels[selectedFeature]]}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke={chartColors[selectedFeature]}
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: chartColors[selectedFeature], strokeWidth: 0 }}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
      <Card title="Score das avaliações" description="Avaliações carregadas associadas às observações pelo observation_id.">
        {joined.length < 2 || joined.every((item) => item.risk === null) ? (
          <EmptyState title="Sem avaliações correspondentes">Solicite avaliações para observações deste episódio.</EmptyState>
        ) : (
          <div className="chart-area">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={joined} margin={{ top: 14, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#29343f" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "#91a0ad", fontSize: 11 }} axisLine={{ stroke: "#34414d" }} tickLine={false} minTickGap={24} />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(value: number) => `${Math.round(value * 100)}%`}
                  tick={{ fill: "#91a0ad", fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={42}
                />
                <Tooltip
                  contentStyle={{ background: "#111820", border: "1px solid #34414d", borderRadius: 10, color: "#eef3f6" }}
                  formatter={(value) => [value == null ? "Sem avaliação" : `${(Number(value) * 100).toFixed(2)}%`, "Score calibrado"]}
                />
                <Line type="monotone" dataKey="risk" stroke="#9d8cff" strokeWidth={2.5} dot={{ r: 3 }} connectNulls={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
      <p className="chart-footnote">Gráficos calculados localmente a partir dos registros carregados. Não representam atualização automática nem monitoramento em tempo real.</p>
    </div>
  );
}
