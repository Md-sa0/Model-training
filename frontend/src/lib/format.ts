import type { FeatureKey } from "../types/api";

export const featureLabels: Record<FeatureKey, string> = {
  HR: "Frequência cardíaca",
  O2Sat: "Saturação de O₂",
  Temp: "Temperatura",
  SBP: "Pressão sistólica",
  MAP: "Pressão arterial média",
  DBP: "Pressão diastólica",
  Resp: "Frequência respiratória",
  WBC: "Leucócitos",
  Lactate: "Lactato",
  Creatinine: "Creatinina",
  Platelets: "Plaquetas",
  Age: "Idade",
  Gender: "Gênero (código do dataset)",
  ICULOS: "Tempo na UTI (ICULOS)",
};

export function formatDate(value: string | null | undefined) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Data inválida";
  return new Intl.DateTimeFormat("pt-BR", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function formatScore(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 2 }).format(value);
}

export function formatFeatureValue(value: number | null | undefined, unit?: string) {
  if (value === null || value === undefined) return "Ausente";
  return `${new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 2 }).format(value)}${unit ? ` ${unit}` : ""}`;
}

export function toIsoWithTimezone(localDateTime: string) {
  const date = new Date(localDateTime);
  if (!localDateTime || Number.isNaN(date.getTime())) return null;
  return date.toISOString();
}

export function describeDecision(decision: string) {
  if (decision === "RISK_ELEVATED") return "Risco elevado (regra operacional)";
  if (decision === "LOW_RISK") return "Baixo risco estimado";
  return decision;
}

export function isUuid(value: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}
