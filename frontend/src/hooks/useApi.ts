import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as api from "../api/endpoints";
import type {
  AssessmentCreateRequest,
  EpisodeCreateRequest,
  ObservationCreateRequest,
  PatientCreateRequest,
} from "../types/api";

export const queryKeys = {
  health: ["health"] as const,
  config: ["config"] as const,
  patients: (offset = 0) => ["patients", 50, offset] as const,
  patient: (id: string) => ["patient", id] as const,
  episodes: (id: string) => ["episodes", id] as const,
  observations: (id: string, episode: string, offset: number) =>
    ["observations", id, episode, 20, offset] as const,
  assessments: (id: string, episode: string, offset: number) =>
    ["assessments", id, episode, 20, offset] as const,
};

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.getHealth,
    retry: false,
    refetchInterval: 30_000,
  });
}

export function useConfig() {
  return useQuery({ queryKey: queryKeys.config, queryFn: api.getConfig, retry: false });
}

export function usePatients(offset = 0) {
  return useQuery({
    queryKey: queryKeys.patients(offset),
    queryFn: () => api.getPatients(50, offset),
    retry: false,
  });
}

export function usePatient(patientId: string) {
  return useQuery({
    queryKey: queryKeys.patient(patientId),
    queryFn: () => api.getPatient(patientId),
    enabled: Boolean(patientId),
    retry: false,
  });
}

export function useEpisodes(patientId: string) {
  return useQuery({
    queryKey: queryKeys.episodes(patientId),
    queryFn: () => api.getEpisodes(patientId),
    enabled: Boolean(patientId),
    retry: false,
  });
}

export function useObservations(patientId: string, episodeId: string, offset: number) {
  return useQuery({
    queryKey: queryKeys.observations(patientId, episodeId, offset),
    queryFn: () => api.getObservations(patientId, episodeId, 20, offset),
    enabled: Boolean(patientId && episodeId),
    retry: false,
  });
}

export function useAssessments(patientId: string, episodeId: string, offset: number) {
  return useQuery({
    queryKey: queryKeys.assessments(patientId, episodeId, offset),
    queryFn: () => api.getAssessments(patientId, episodeId, 20, offset),
    enabled: Boolean(patientId && episodeId),
    retry: false,
  });
}

export function useCreatePatient() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: PatientCreateRequest) => api.createPatient(payload),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["patients"] });
    },
  });
}

export function useCreateEpisode(patientId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: EpisodeCreateRequest) => api.createEpisode(patientId, payload),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: queryKeys.patient(patientId) }),
        client.invalidateQueries({ queryKey: queryKeys.episodes(patientId) }),
      ]);
    },
  });
}

export function useCreateObservation(patientId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: ObservationCreateRequest) => api.createObservation(patientId, payload),
    onSuccess: async (_observation, payload) => {
      await client.invalidateQueries({
        queryKey: ["observations", patientId, payload.episode_id],
      });
    },
  });
}

export function useCreateAssessment(patientId: string, episodeId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: AssessmentCreateRequest) => api.createAssessment(patientId, payload),
    onSuccess: async () => {
      await client.invalidateQueries({
        queryKey: ["assessments", patientId, episodeId],
      });
    },
  });
}
