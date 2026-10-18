import {
  ArrowLeft,
  CalendarClock,
  ClipboardPlus,
  FilePlus2,
  Plus,
  RefreshCw,
} from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { toApiError } from "../api/client";
import { DisclaimerBanner } from "../components/DisclaimerBanner";
import { EpisodeForm } from "../components/EpisodeForm";
import { ObservationCharts } from "../components/ObservationCharts";
import { ObservationForm } from "../components/ObservationForm";
import {
  Alert,
  Button,
  Card,
  EmptyState,
  Field,
  LoadingState,
  Modal,
  Pagination,
  SelectInput,
} from "../components/ui";
import {
  useAssessments,
  useCreateAssessment,
  useEpisodes,
  useObservations,
  usePatient,
} from "../hooks/useApi";
import { describeDecision, formatDate, formatFeatureValue, formatScore } from "../lib/format";
import type { Assessment, Episode, Observation } from "../types/api";
import { featureKeys } from "../types/api";

const PAGE_SIZE = 20;

function EpisodeDialog({
  patientId,
  onCreated,
}: {
  patientId: string;
  onCreated: (episodeId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button variant="secondary" onClick={() => setOpen(true)}><Plus size={16} /> Nova internação</Button>
      <Modal open={open} onOpenChange={setOpen} title="Criar episódio" description="Use um episódio separado para cada internação ou readmissão.">
        <EpisodeForm patientId={patientId} onCreated={(id) => { onCreated(id); setOpen(false); }} />
      </Modal>
    </>
  );
}

function ObservationDialog({
  patientId,
  episodes,
  selectedEpisodeId,
  onSaved,
}: {
  patientId: string;
  episodes: Episode[];
  selectedEpisodeId: string;
  onSaved: (observation: Observation) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button onClick={() => setOpen(true)} disabled={!episodes.length}><FilePlus2 size={16} /> Registrar observação</Button>
      <Modal
        open={open}
        onOpenChange={setOpen}
        title="Nova observação clínica"
        description="O registro não executa inferência. Os campos são enviados sem transformação de unidade."
      >
        <ObservationForm
          patientId={patientId}
          episodes={episodes}
          selectedEpisodeId={selectedEpisodeId}
          onSaved={(observation) => { onSaved(observation); setOpen(false); }}
        />
      </Modal>
    </>
  );
}

function ObservationHistoryTable({
  observations,
  assessments,
  isLoading,
  isError,
  errorMessage,
  offset,
  total,
  onPageChange,
  onRetry,
  onAssess,
  pending,
}: {
  observations: Observation[] | undefined;
  assessments: Assessment[] | undefined;
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
  offset: number;
  total: number | undefined;
  onPageChange: (offset: number) => void;
  onRetry: () => void;
  onAssess: (observationId: string) => void;
  pending: boolean;
}) {
  if (isLoading) return <LoadingState label="Carregando observações..." />;
  if (isError) return <Alert tone="error" title="Falha ao carregar observações">{errorMessage}<div><Button variant="secondary" onClick={onRetry}>Tentar novamente</Button></div></Alert>;
  if (!observations?.length) return <EmptyState title="Nenhuma observação neste episódio">Registre uma observação para começar o histórico.</EmptyState>;

  const assessed = new Set((assessments ?? []).map((item) => item.observation_id));
  return (
    <>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Data da observação</th><th>Identificador</th><th>Valores recebidos</th><th>Avaliação</th></tr></thead>
          <tbody>
            {observations.map((observation) => {
              const visibleValues = featureKeys.filter((feature) => observation.values[feature] !== undefined);
              return (
                <tr key={observation.observation_id}>
                  <td>{formatDate(observation.observed_at)}</td>
                  <td><span className="mono-id">{observation.observation_id}</span></td>
                  <td>
                    <details className="value-details">
                      <summary>{visibleValues.length} campo(s) enviado(s) · {observation.missing_fields.length} ausente(s)</summary>
                      <div className="value-grid">
                        {featureKeys.map((feature) => (
                          <span key={feature}><b>{feature}</b> {formatFeatureValue(observation.values[feature], observation.units[feature])}</span>
                        ))}
                      </div>
                    </details>
                  </td>
                  <td>
                    <Button
                      variant="secondary"
                      className="small-button"
                      disabled={pending}
                      onClick={() => onAssess(observation.observation_id)}
                      aria-label={`${assessed.has(observation.observation_id) ? "Consultar" : "Solicitar"} avaliação para observação ${observation.observation_id}`}
                    >
                      <ClipboardPlus size={15} /> {assessed.has(observation.observation_id) ? "Consultar avaliação" : "Solicitar avaliação"}
                    </Button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Pagination offset={offset} total={total ?? 0} limit={PAGE_SIZE} onChange={onPageChange} />
    </>
  );
}

function AssessmentHistoryTable({
  assessments,
  observations,
  isLoading,
  isError,
  errorMessage,
  offset,
  total,
  onPageChange,
  onRetry,
}: {
  assessments: Assessment[] | undefined;
  observations: Observation[] | undefined;
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
  offset: number;
  total: number | undefined;
  onPageChange: (offset: number) => void;
  onRetry: () => void;
}) {
  if (isLoading) return <LoadingState label="Carregando avaliações..." />;
  if (isError) return <Alert tone="error" title="Falha ao carregar avaliações">{errorMessage}<div><Button variant="secondary" onClick={onRetry}>Tentar novamente</Button></div></Alert>;
  if (!assessments?.length) return <EmptyState title="Nenhuma avaliação neste episódio">Solicite uma avaliação explicitamente na linha de uma observação.</EmptyState>;

  const observationDates = new Map((observations ?? []).map((item) => [item.observation_id, item.observed_at]));
  return (
    <>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Observação</th><th>Horário observado</th><th>Avaliado em</th><th>Score calibrado</th><th>Decisão operacional</th><th>Configuração</th></tr></thead>
          <tbody>
            {assessments.map((assessment) => (
              <tr key={assessment.assessment_id}>
                <td><span className="mono-id">{assessment.observation_id}</span></td>
                <td>{formatDate(observationDates.get(assessment.observation_id))}</td>
                <td>{formatDate(assessment.evaluated_at)}</td>
                <td><strong>{formatScore(assessment.risk_score)}</strong><small className="table-subline">limiar {formatScore(assessment.threshold)}</small></td>
                <td><span className={`decision-pill ${assessment.decision === "RISK_ELEVATED" ? "decision-elevated" : "decision-low"}`}>{describeDecision(assessment.decision)}</span></td>
                <td><span className="version-tag">{assessment.calibration_method}</span><small className="table-subline">{assessment.config_version.slice(0, 12)}…</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pagination offset={offset} total={total ?? 0} limit={PAGE_SIZE} onChange={onPageChange} />
    </>
  );
}

function LatestAssessment({
  assessment,
  observation,
  onRetry,
  retryPending,
}: {
  assessment: Assessment;
  observation?: Observation;
  onRetry: () => void;
  retryPending: boolean;
}) {
  return (
    <Card title="Resultado da avaliação" description={`Observação vinculada: ${observation ? formatDate(observation.observed_at) : assessment.observation_id}`}>
      <div className="assessment-summary">
        <div><span>Decisão</span><strong className={`decision-text ${assessment.decision === "RISK_ELEVATED" ? "decision-text-elevated" : ""}`}>{describeDecision(assessment.decision)}</strong></div>
        <div><span>Score calibrado</span><strong>{formatScore(assessment.risk_score)}</strong></div>
        <div><span>Limiar congelado</span><strong>{formatScore(assessment.threshold)}</strong></div>
        <div><span>Método</span><strong>{assessment.calibration_method}</strong></div>
      </div>
      <p className="assessment-message">{assessment.message}</p>
      <p className="subtle-copy">{assessment.risk_score_meaning}</p>
      {assessment.imputed_features.length > 0 && (
        <Alert tone="warning" title="Features dinâmicas imputadas">
          {assessment.imputed_features.join(", ")}
        </Alert>
      )}
      <details className="version-details">
        <summary>Detalhes e rastreabilidade</summary>
        <dl>
          <dt>Avaliado em</dt><dd>{formatDate(assessment.evaluated_at)}</dd>
          <dt>Versão do modelo</dt><dd className="mono-id">{assessment.model_version}</dd>
          <dt>Versão da configuração</dt><dd className="mono-id">{assessment.config_version}</dd>
          <dt>Score bruto</dt><dd>{assessment.raw_score.toFixed(6)}</dd>
          <dt>Identificador da avaliação</dt><dd className="mono-id">{assessment.assessment_id}</dd>
        </dl>
      </details>
      <Alert tone="info">{assessment.disclaimer}</Alert>
      <Button variant="ghost" disabled={retryPending} onClick={onRetry}>
        <RefreshCw size={15} /> Solicitar/consultar novamente
      </Button>
    </Card>
  );
}

export function PatientDetailPage() {
  const { patientId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const [observationOffset, setObservationOffset] = useState(0);
  const [assessmentOffset, setAssessmentOffset] = useState(0);
  const [lastAssessmentObservationId, setLastAssessmentObservationId] = useState<string | null>(null);
  const [latestAssessment, setLatestAssessment] = useState<Assessment | null>(null);
  const patient = usePatient(patientId);
  const episodesQuery = useEpisodes(patientId);
  const episodes = episodesQuery.data ?? patient.data?.episodes ?? [];
  const requestedEpisodeId = searchParams.get("episode_id") ?? "";
  const selectedEpisodeId = episodes.some((item) => item.episode_id === requestedEpisodeId)
    ? requestedEpisodeId
    : episodes[0]?.episode_id ?? "";
  const observations = useObservations(patientId, selectedEpisodeId, observationOffset);
  const assessments = useAssessments(patientId, selectedEpisodeId, assessmentOffset);
  const createAssessment = useCreateAssessment(patientId, selectedEpisodeId);

  const selectedAssessmentObservation = useMemo(
    () => observations.data?.items.find((item) => item.observation_id === lastAssessmentObservationId),
    [observations.data?.items, lastAssessmentObservationId],
  );
  const patientError = patient.error ? toApiError(patient.error) : null;
  const observationError = observations.error ? toApiError(observations.error) : null;
  const assessmentError = assessments.error ? toApiError(assessments.error) : null;
  const mutationError = createAssessment.error ? toApiError(createAssessment.error) : null;

  async function requestAssessment(observationId: string) {
    setLastAssessmentObservationId(observationId);
    try {
      const result = await createAssessment.mutateAsync({
        assessment_id: crypto.randomUUID(),
        observation_id: observationId,
      });
      setLatestAssessment(result);
    } catch {
      // Mutation errors are rendered below.
    }
  }

  function changeEpisode(episodeId: string) {
    setSearchParams({ episode_id: episodeId });
    setObservationOffset(0);
    setAssessmentOffset(0);
    setLatestAssessment(null);
  }

  function handleSaved(observation: Observation) {
    setSearchParams({ episode_id: observation.episode_id });
    setObservationOffset(0);
    setAssessmentOffset(0);
  }

  function handleEpisodeCreated(episodeId: string) {
    changeEpisode(episodeId);
  }

  if (patient.isPending) return <LoadingState label="Carregando dados do paciente..." />;
  if (patient.isError || !patient.data) {
    return (
      <div className="page-stack">
        <Link className="back-link" to="/patients"><ArrowLeft size={16} /> Lista de pacientes</Link>
        <Alert tone="error" title={patientError?.status === 404 ? "Paciente não encontrado" : "Não foi possível carregar o paciente"}>
          {patientError?.message ?? "Verifique a conexão com a API."}
          <div><Button variant="secondary" onClick={() => void patient.refetch()}>Tentar novamente</Button></div>
        </Alert>
      </div>
    );
  }

  const currentEpisode = episodes.find((item) => item.episode_id === selectedEpisodeId);
  const latestObservation = observations.data?.items[0];

  return (
    <div className="page-stack">
      <div className="detail-breadcrumb">
        <Link className="back-link" to="/patients"><ArrowLeft size={16} /> Pacientes</Link>
        <span>/</span><span>Detalhes</span>
      </div>
      <DisclaimerBanner />
      <Card
        title="Identificação do paciente"
        description="Dados identificadores retornados pela API local."
        action={<EpisodeDialog patientId={patientId} onCreated={handleEpisodeCreated} />}
      >
        <div className="patient-identity-grid">
          <div><span>ID interno</span><strong className="mono-id">{patient.data.patient_id}</strong></div>
          <div><span>ID externo</span><strong>{patient.data.external_id || "Não informado"}</strong></div>
          <div><span>Cadastrado em</span><strong>{formatDate(patient.data.created_at)}</strong></div>
        </div>
        {episodesQuery.isError && (
          <Alert tone="warning" title="Lista de episódios não atualizada">
            {toApiError(episodesQuery.error).message}
            <Button variant="ghost" onClick={() => void episodesQuery.refetch()}>Tentar novamente</Button>
          </Alert>
        )}
        <div className="episode-selector">
          <Field label="Episódio ativo para consulta" htmlFor="episode-picker">
            <SelectInput id="episode-picker" value={selectedEpisodeId} onChange={(event) => changeEpisode(event.target.value)}>
              {episodes.map((episode) => (
                <option key={episode.episode_id} value={episode.episode_id}>
                  {episode.label || "Episódio sem rótulo"} · {formatDate(episode.started_at)}
                </option>
              ))}
            </SelectInput>
          </Field>
          <span className="episode-id"><CalendarClock size={15} /> {currentEpisode?.episode_id ?? "Nenhum episódio"}</span>
        </div>
      </Card>

      {!selectedEpisodeId ? (
        <Alert tone="warning" title="Nenhum episódio disponível">Crie um episódio para registrar observações.</Alert>
      ) : (
        <>
          <Card
            title="Histórico de observações"
            description="Registros imutáveis associados exclusivamente ao episódio selecionado."
            action={<ObservationDialog patientId={patientId} episodes={episodes} selectedEpisodeId={selectedEpisodeId} onSaved={handleSaved} />}
          >
            <ObservationHistoryTable
              observations={observations.data?.items}
              assessments={assessments.data?.items}
              isLoading={observations.isPending}
              isError={observations.isError}
              errorMessage={observationError?.message}
              offset={observationOffset}
              total={observations.data?.total}
              onPageChange={setObservationOffset}
              onRetry={() => void observations.refetch()}
              onAssess={(id) => void requestAssessment(id)}
              pending={createAssessment.isPending}
            />
          </Card>

          {mutationError && (
            <Alert tone="error" title={mutationError.status === 503 ? "Modelo indisponível para inferência" : "Não foi possível obter a avaliação"}>
              {mutationError.status === 503
                ? "A API está acessível, mas o modelo não está pronto (503 MODEL_NOT_READY). Verifique o estado no cabeçalho e tente novamente quando estiver pronto."
                : mutationError.message}
              {mutationError.validation.length > 0 && <ul>{mutationError.validation.map((item) => <li key={item}>{item}</li>)}</ul>}
              {lastAssessmentObservationId && (
                <Button variant="secondary" disabled={createAssessment.isPending} onClick={() => void requestAssessment(lastAssessmentObservationId)}>
                  Tentar novamente
                </Button>
              )}
            </Alert>
          )}

          {latestAssessment && (
            <LatestAssessment
              assessment={latestAssessment}
              observation={selectedAssessmentObservation ?? latestObservation}
              onRetry={() => void requestAssessment(latestAssessment.observation_id)}
              retryPending={createAssessment.isPending}
            />
          )}

          <ObservationCharts
            observations={observations.data?.items ?? []}
            assessments={assessments.data?.items ?? []}
          />

          <Card title="Histórico de avaliações" description="Avaliações persistidas, com referência à observação e configuração utilizadas.">
            <AssessmentHistoryTable
              assessments={assessments.data?.items}
              observations={observations.data?.items}
              isLoading={assessments.isPending}
              isError={assessments.isError}
              errorMessage={assessmentError?.message}
              offset={assessmentOffset}
              total={assessments.data?.total}
              onPageChange={setAssessmentOffset}
              onRetry={() => void assessments.refetch()}
            />
          </Card>
        </>
      )}
    </div>
  );
}
