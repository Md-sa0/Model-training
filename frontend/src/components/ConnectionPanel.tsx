import { AlertTriangle, CheckCircle2, CircleHelp, Cpu, Server } from "lucide-react";
import { toApiError } from "../api/client";
import { useConfig, useHealth } from "../hooks/useApi";
import { Alert, Button, Card, LoadingState } from "./ui";

export function ConnectionPanel() {
  const health = useHealth();
  const config = useConfig();
  const isApiAvailable = health.isSuccess;
  const modelReady = isApiAvailable && health.data.model_ready && config.data?.model_ready === true;

  if (health.isPending || config.isPending) {
    return <Card title="Estado do serviço" description="Verificando conexão e prontidão do modelo."><LoadingState /></Card>;
  }

  if (health.isError) {
    const error = toApiError(health.error);
    return (
      <Card title="Estado do serviço" description="A API e a disponibilidade de inferência são verificações distintas.">
        <Alert tone="error" title="API indisponível">{error.message}</Alert>
        <Button className="retry-button" variant="secondary" onClick={() => void health.refetch()}>Tentar novamente</Button>
      </Card>
    );
  }

  const configError = config.isError ? toApiError(config.error) : null;
  return (
    <Card title="Estado do serviço" description="API acessível não significa necessariamente que a inferência esteja pronta.">
      <div className="status-grid">
        <div className="status-item">
          <span className={`status-icon ${isApiAvailable ? "status-icon-ok" : ""}`}><Server size={18} /></span>
          <div><small>API</small><strong>{isApiAvailable ? "Acessível" : "Indisponível"}</strong></div>
          {isApiAvailable ? <CheckCircle2 className="status-check" size={17} /> : <AlertTriangle size={17} />}
        </div>
        <div className="status-item">
          <span className={`status-icon ${modelReady ? "status-icon-ok" : "status-icon-warn"}`}><Cpu size={18} /></span>
          <div><small>Modelo</small><strong>{modelReady ? "Pronto para inferência" : "Indisponível"}</strong></div>
          {modelReady ? <CheckCircle2 className="status-check" size={17} /> : <AlertTriangle className="status-warn" size={17} />}
        </div>
      </div>
      {!modelReady && (
        <Alert tone="warning" title="Avaliações não disponíveis">
          O serviço está respondendo, mas não confirmou a prontidão do modelo. A solicitação de avaliação poderá retornar
          <code>503 MODEL_NOT_READY</code>.
          {health.data.model_error && <span> Falha de inicialização: {health.data.model_error}.</span>}
        </Alert>
      )}
      {configError && (
        <Alert tone="warning" title="Configuração não consultada">{configError.message}</Alert>
      )}
      {config.data?.model_ready && (
        <div className="config-strip">
          <span><CircleHelp size={15} /> Calibrador <strong>{config.data.calibration_method}</strong></span>
          <span>Limiar congelado <strong>{config.data.threshold}</strong></span>
          {config.data.threshold_target_met === false && <span className="target-not-met">Meta de sensibilidade não atingida</span>}
        </div>
      )}
      <p className="subtle-copy">O serviço não recebe dados automaticamente e não monitora pacientes em tempo real.</p>
    </Card>
  );
}
