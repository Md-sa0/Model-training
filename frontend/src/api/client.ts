import axios, { AxiosError } from "axios";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const apiClient = axios.create({
  baseURL: import.meta.env.DEV ? "" : configuredBaseUrl,
  timeout: 120_000,
  headers: { "Content-Type": "application/json" },
});

export interface ApiError {
  status: number | null;
  message: string;
  code: string | null;
  validation: string[];
}

export function toApiError(error: unknown): ApiError {
  if (!axios.isAxiosError(error)) {
    return { status: null, message: "Ocorreu um erro inesperado.", code: null, validation: [] };
  }

  const axiosError = error as AxiosError<{ detail?: unknown }>;
  const detail = axiosError.response?.data?.detail;
  const validation = Array.isArray(detail)
    ? detail.map((item) => {
        if (typeof item === "object" && item !== null && "msg" in item) {
          const location = "loc" in item && Array.isArray(item.loc) ? item.loc.slice(1).join(" → ") : "";
          return location ? `${location}: ${String(item.msg)}` : String(item.msg);
        }
        return String(item);
      })
    : [];

  let message = "Não foi possível concluir a solicitação.";
  let code: string | null = null;
  if (typeof detail === "string") message = detail;
  else if (detail && typeof detail === "object") {
    if ("message" in detail && typeof detail.message === "string") message = detail.message;
    if ("code" in detail && typeof detail.code === "string") code = detail.code;
  } else if (!axiosError.response) {
    message = "Não foi possível conectar à API. Verifique se o backend está em execução.";
  }

  if (axiosError.response?.status === 409 && !message) {
    message = "O registro conflita com um identificador ou horário já existente.";
  }
  return {
    status: axiosError.response?.status ?? null,
    message,
    code,
    validation,
  };
}
