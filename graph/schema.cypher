// One node per drug, condition and patient.
// WHY: a uniqueness constraint stops a second load from creating duplicates,
// and it also builds an index, so lookups by name or id are fast.
CREATE CONSTRAINT drug_name IF NOT EXISTS
FOR (d:Drug) REQUIRE d.name IS UNIQUE;

CREATE CONSTRAINT condition_name IF NOT EXISTS
FOR (c:Condition) REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT patient_id IF NOT EXISTS
FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE;
