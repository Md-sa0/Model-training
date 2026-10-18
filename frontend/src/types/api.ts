export const featureKeys = [
  "HR",
  "O2Sat",
  "Temp",
  "SBP",
  "MAP",
  "DBP",
  "Resp",
  "WBC",
  "Lactate",
  "Creatinine",
  "Platelets",
  "Age",
  "Gender",
  "ICULOS",
] as const;

export type FeatureKey = (typeof featureKeys)[number];
export type ClinicalValue = number | null;

export interface Health {
  status: string;
  model_ready: boolean;
  model_error: string | null;
}

export interface ApiConfig {
  model_ready: boolean;
  model_id: string | null;
  model_version: string | null;
  config_version: string | null;
  calibration_method: string | null;
  calibrator_locked: boolean | null;
  threshold: number | null;
  threshold_locked: boolean | null;
  threshold_target_met: boolean | null;
  fitted_on: string | null;
  test_used_for_fitting: boolean | null;
  features: FeatureKey[];
  dynamic_features: FeatureKey[];
  decision_semantics: string;
  risk_score_meaning: string;
  disclaimer: string;
}

export interface Patient {
  patient_id: string;
  external_id: string | null;
  created_at: string;
}

export interface Episode {
  episode_id: string;
  patient_id: string;
  label: string | null;
  started_at: string;
  created_at: string;
}

export interface PatientCreated {
  patient: Patient;
  episode: Episode;
}

export interface PatientDetail extends Patient {
  episodes: Episode[];
}

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Observation {
  observation_id: string;
  patient_id: string;
  episode_id: string;
  observed_at: string;
  received_at: string;
  values: Partial<Record<FeatureKey, ClinicalValue>>;
  units: Partial<Record<FeatureKey, string>>;
  missing_fields: FeatureKey[];
}

export interface Assessment {
  assessment_id: string;
  observation_id: string;
  patient_id: string;
  episode_id: string;
  evaluated_at: string;
  raw_score: number;
  risk_score: number;
  risk_score_meaning: string;
  threshold: number;
  decision: "RISK_ELEVATED" | "LOW_RISK" | string;
  calibration_method: string;
  model_version: string;
  config_version: string;
  imputed_features: FeatureKey[];
  message: string;
  disclaimer: string;
}

export interface PatientCreateRequest {
  external_id?: string | null;
  initial_episode_label?: string | null;
  episode_started_at?: string | null;
}

export interface EpisodeCreateRequest {
  label?: string | null;
  started_at?: string | null;
}

export interface ObservationCreateRequest {
  observation_id: string;
  episode_id: string;
  observed_at: string;
  values: Record<FeatureKey, ClinicalValue>;
  units?: Partial<Record<FeatureKey, string>>;
}

export interface AssessmentCreateRequest {
  assessment_id: string;
  observation_id: string;
}

export interface PatientRiskRequest {
  values: Partial<Record<FeatureKey, ClinicalValue>>;
}

export type PatientRiskResponse = Omit<
  Assessment,
  "assessment_id" | "observation_id" | "patient_id" | "episode_id" | "evaluated_at"
>;
