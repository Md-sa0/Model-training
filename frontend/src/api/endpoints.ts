import { apiClient } from "./client";
import type {
  ApiConfig,
  Assessment,
  AssessmentCreateRequest,
  Episode,
  EpisodeCreateRequest,
  Health,
  Observation,
  ObservationCreateRequest,
  Page,
  Patient,
  PatientCreateRequest,
  PatientCreated,
  PatientDetail,
} from "../types/api";

export async function getHealth() {
  return (await apiClient.get<Health>("/health")).data;
}

export async function getConfig() {
  return (await apiClient.get<ApiConfig>("/api/config")).data;
}

export async function getPatients(limit = 50, offset = 0) {
  return (await apiClient.get<Page<Patient>>("/api/patients", { params: { limit, offset } })).data;
}

export async function createPatient(payload: PatientCreateRequest) {
  return (await apiClient.post<PatientCreated>("/api/patients", payload)).data;
}

export async function getPatient(patientId: string) {
  return (await apiClient.get<PatientDetail>(`/api/patients/${patientId}`)).data;
}

export async function getEpisodes(patientId: string) {
  return (await apiClient.get<Episode[]>(`/api/patients/${patientId}/episodes`)).data;
}

export async function createEpisode(patientId: string, payload: EpisodeCreateRequest) {
  return (await apiClient.post<Episode>(`/api/patients/${patientId}/episodes`, payload)).data;
}

export async function getObservations(patientId: string, episodeId: string, limit = 50, offset = 0) {
  return (
    await apiClient.get<Page<Observation>>(`/api/patients/${patientId}/observations`, {
      params: { episode_id: episodeId, limit, offset },
    })
  ).data;
}

export async function createObservation(patientId: string, payload: ObservationCreateRequest) {
  return (
    await apiClient.post<Observation>(`/api/patients/${patientId}/observations`, payload)
  ).data;
}

export async function getAssessments(patientId: string, episodeId: string, limit = 50, offset = 0) {
  return (
    await apiClient.get<Page<Assessment>>(`/api/patients/${patientId}/assessments`, {
      params: { episode_id: episodeId, limit, offset },
    })
  ).data;
}

export async function createAssessment(
  patientId: string,
  payload: AssessmentCreateRequest,
) {
  return (
    await apiClient.post<Assessment>(`/api/patients/${patientId}/assessments`, payload)
  ).data;
}
