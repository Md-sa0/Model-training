import { ArrowUpRight, Users } from "lucide-react";
import { Link } from "react-router-dom";
import type { Patient } from "../types/api";
import { formatDate } from "../lib/format";
import { Button, EmptyState, LoadingState, Pagination } from "./ui";

export function PatientsTable({
  patients,
  total,
  offset,
  isLoading,
  isError,
  errorMessage,
  onPageChange,
  onRetry,
}: {
  patients: Patient[] | undefined;
  total: number | undefined;
  offset: number;
  isLoading: boolean;
  isError: boolean;
  errorMessage?: string;
  onPageChange: (offset: number) => void;
  onRetry: () => void;
}) {
  if (isLoading) return <LoadingState label="Carregando pacientes cadastrados..." />;
  if (isError) {
    return (
      <div className="state-box state-error">
        <span>{errorMessage || "Não foi possível consultar os pacientes."}</span>
        <Button variant="secondary" onClick={onRetry}>Tentar novamente</Button>
      </div>
    );
  }
  if (!patients?.length) return <EmptyState title="Nenhum paciente cadastrado">Cadastre um paciente para iniciar um registro longitudinal.</EmptyState>;

  return (
    <>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Paciente</th><th>Identificador externo</th><th>Cadastro</th><th><span className="sr-only">Ação</span></th></tr></thead>
          <tbody>
            {patients.map((patient) => (
              <tr key={patient.patient_id}>
                <td>
                  <Link className="patient-cell-link" to={`/patients/${patient.patient_id}`}>
                    <span className="avatar"><Users size={16} /></span>
                    <span><strong>Paciente</strong><small>{patient.patient_id}</small></span>
                  </Link>
                </td>
                <td>{patient.external_id || <span className="subtle-copy">Não informado</span>}</td>
                <td>{formatDate(patient.created_at)}</td>
                <td><Link className="row-action" to={`/patients/${patient.patient_id}`} aria-label="Abrir paciente"><ArrowUpRight size={17} /></Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pagination offset={offset} total={total ?? 0} limit={50} onChange={onPageChange} />
    </>
  );
}
