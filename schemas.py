from typing import Optional
from pydantic import BaseModel, Field


class DenialExtraction(BaseModel):
    patient_name: Optional[str] = Field(default=None)
    patient_account_number: Optional[str] = Field(default=None)
    service_date_start: Optional[str] = Field(default=None)
    service_date_end: Optional[str] = Field(default=None)

    denial_type: Optional[str] = Field(
        default=None,
        description="Examples: DRG downgrade, clinical validation denial, medical necessity denial, coding denial, level of care denial, authorization denial, timely filing denial."
    )

    before_value: Optional[str] = Field(
        default=None,
        description="Original billed/requested DRG, diagnosis, procedure, level of care, or value being denied."
    )

    after_value: Optional[str] = Field(
        default=None,
        description="Payer revised/approved/recommended DRG, diagnosis, procedure, level of care, or replacement value."
    )

    drg_before_value: Optional[str] = Field(
        default=None,
        description="Original/billed/requested MS-DRG or DRG value before payer review, if clearly stated."
    )

    drg_after_value: Optional[str] = Field(
        default=None,
        description="Payer-recommended/revised/approved MS-DRG or DRG value after review, if clearly stated."
    )

    policy_type: Optional[str] = Field(
        default=None,
        description="Examples: Medicare, Medicaid, Commercial, Medicare Advantage, Managed Medicaid."
    )

    provider_name: Optional[str] = Field(
        default=None,
        description="Payer or reviewing organization name, such as Humana, Aetna, Optum, Cotiviti."
    )

    claim_number: Optional[str] = Field(default=None)

    summary: Optional[str] = Field(
        default=None,
        description="Plain-English summary of what was denied."
    )