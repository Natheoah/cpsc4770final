# questions.py
#
# Built-in sample of 10 text-only HLE questions used as a fallback when the
# real dataset isn't available. To use all 2,500 real questions run:
#
#   HF_TOKEN=<token> python3 fetch_questions.py
#
# That writes hle_dataset.json which run_eval.py loads automatically when
# you pass --questions hle_dataset.json.

SAMPLE_QUESTIONS = [
    {
        "id": "hle_001",
        "subject": "Mathematics",
        "question": (
            "Let p(x) be a monic polynomial of degree 4 with real coefficients "
            "and two distinct real roots r1 and r2, each of multiplicity 2. "
            "If p(0) = 9 and p(1) = 16, find r1 + r2."
        ),
        "answer": "1",
        "answer_type": "exact",
    },
    {
        "id": "hle_002",
        "subject": "Physics",
        "question": (
            "A particle of mass m is confined to move on the surface of a sphere "
            "of radius R. The particle is in a quantum state described by the "
            "spherical harmonic Y_2^1(theta, phi). What is the expectation value "
            "of L_z (the z-component of angular momentum) in units of hbar?"
        ),
        "answer": "1",
        "answer_type": "exact",
    },
    {
        "id": "hle_003",
        "subject": "Chemistry",
        "question": (
            "In the Diels-Alder reaction between (E)-1-methoxybuta-1,3-diene and "
            "maleic anhydride, what is the major product's stereochemistry at the "
            "newly formed stereocenters? "
            "Options: (A) endo  (B) exo  (C) syn  (D) anti"
        ),
        "answer": "A",
        "answer_type": "multiple_choice",
    },
    {
        "id": "hle_004",
        "subject": "Mathematics",
        "question": "How many non-isomorphic groups of order 16 exist?",
        "answer": "14",
        "answer_type": "exact",
    },
    {
        "id": "hle_005",
        "subject": "Computer Science",
        "question": (
            "In a skip list with n elements, what is the expected number of nodes "
            "examined during a search operation, expressed in asymptotic notation?"
        ),
        "answer": "O(log n)",
        "answer_type": "exact",
    },
    {
        "id": "hle_006",
        "subject": "Biology",
        "question": (
            "Which enzyme removes RNA primers during DNA replication in E. coli "
            "and replaces them with DNA?"
        ),
        "answer": "DNA Polymerase I",
        "answer_type": "exact",
    },
    {
        "id": "hle_007",
        "subject": "Mathematics",
        "question": (
            "What is the value of the integral from 0 to infinity of "
            "(sin x / x)^2 dx?"
        ),
        "answer": "pi/2",
        "answer_type": "exact",
    },
    {
        "id": "hle_008",
        "subject": "Philosophy",
        "question": (
            "In Kripke's possible worlds semantics for modal logic, what condition "
            "must hold for a proposition to be necessarily true?"
        ),
        "answer": "It must be true in all accessible possible worlds",
        "answer_type": "exact",
    },
    {
        "id": "hle_009",
        "subject": "Mathematics",
        "question": (
            "Given that 2023 = 7 x 17^2, and that every group of order p*q^2 for "
            "primes p < q with p not dividing (q^2 - 1) is abelian, is every group "
            "of order 2023 abelian? Answer yes or no."
        ),
        "answer": "yes",
        "answer_type": "exact",
    },
    {
        "id": "hle_010",
        "subject": "Physics",
        "question": (
            "In quantum field theory, what is the one-loop correction to the "
            "electron anomalous magnetic moment (g-2)/2, expressed in units of "
            "alpha/(2*pi)?"
        ),
        "answer": "1",
        "answer_type": "exact",
    },
]
