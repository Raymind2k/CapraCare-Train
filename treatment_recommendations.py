"""
CapraCare - Treatment Recommendation Engine
Structured knowledge base for 8 goat skin diseases.
Sources: FAO, WOAH, Merck Veterinary Manual.
"""

DISEASE_DB = {
    "parasitic_mange": {
        "common_name": "Mange (Sarcoptic/Psoroptic)",
        "urgency": "HIGH",
        "contagious": True,
        "zoonotic": True,
        "clinical_signs": [
            "Intense itching, scratching against objects",
            "Hair loss in patches (face, ears, neck, legs)",
            "Thickened, crusty, scaly skin",
            "Secondary bacterial infection from scratching",
        ],
        "immediate_actions": [
            "Isolate affected goat from herd immediately",
            "Wear gloves when handling (zoonotic risk)",
            "Apply topical acaricide (ivermectin pour-on or amitraz dip) — consult vet for dosage",
            "Trim long hair around lesions for better treatment penetration",
        ],
        "management": [
            "Inject ivermectin subcutaneously 200 mcg/kg, repeat after 14 days",
            "Treat all in-contact animals (mange spreads rapidly)",
            "Disinfect housing, feeders, and grooming equipment",
            "Apply soothing emollients to crusted areas after acaricide treatment",
        ],
        "prevention": [
            "Quarantine new animals for 30 days",
            "Regular inspection of skin during grooming",
            "Avoid overcrowding",
            "Routine deworming program",
        ],
        "when_to_call_vet": "If lesions cover >30% of body, secondary infection visible, or no improvement after 14 days",
        "differential_diagnoses": ["dermatophilosis", "lice infestation", "ringworm"],
    },
    "parasitic_myiasis": {
        "common_name": "Myiasis (Maggot Infestation)",
        "urgency": "CRITICAL",
        "contagious": False,
        "zoonotic": False,
        "clinical_signs": [
            "Visible maggots in open wounds",
            "Foul-smelling discharge",
            "Restlessness, off-feed",
            "Rapid tissue destruction",
        ],
        "immediate_actions": [
            "Remove all visible maggots with forceps (clean wound thoroughly)",
            "Flush wound with antiseptic solution (povidone-iodine or chlorhexidine)",
            "Apply maggot-killing spray (cypermethrin-based)",
            "Cover wound with fly-proof dressing",
        ],
        "management": [
            "Daily wound cleaning and dressing changes",
            "Systemic antibiotics if secondary infection suspected",
            "Anti-inflammatory drugs for pain management",
            "Keep animal in fly-free environment until healed",
        ],
        "prevention": [
            "Treat all wounds promptly (even minor cuts)",
            "Fly control measures (traps, repellents)",
            "Tail docking in fly-prone seasons (where appropriate)",
            "Regular shearing in heavily fleeced breeds",
        ],
        "when_to_call_vet": "ALWAYS — myiasis can be fatal within days without proper treatment",
        "differential_diagnoses": ["bacterial wound infection", "abscess"],
    },
    "skin_infection_abscess": {
        "common_name": "Abscess (often Caseous Lymphadenitis)",
        "urgency": "MODERATE",
        "contagious": True,
        "zoonotic": True,
        "clinical_signs": [
            "Firm or fluctuant swelling under skin",
            "Common locations: jaw, neck, shoulders, flank",
            "Eventually ruptures with thick, cheesy pus",
            "Affected lymph nodes enlarged",
        ],
        "immediate_actions": [
            "Isolate animal (CLA is highly contagious to other goats)",
            "DO NOT lance until mature (firm, walled-off)",
            "Wear gloves and PPE (zoonotic)",
            "Once ruptured, clean discharge carefully, dispose of contaminated material by burning",
        ],
        "management": [
            "Mature abscess: lance with sterile blade, drain completely, flush with iodine",
            "Pack wound with antiseptic until healed from inside",
            "Systemic antibiotics ineffective once walled off — focus on local treatment",
            "Cull chronically affected animals to prevent herd spread",
        ],
        "prevention": [
            "Vaccination programs in CLA-endemic areas",
            "Quarantine and inspect new arrivals",
            "Avoid sharing needles, tattoo equipment, shears between animals",
            "Maintain low-injury environment (no sharp edges)",
        ],
        "when_to_call_vet": "For diagnosis confirmation (CLA vs other abscesses), and to advise on herd-level management",
        "differential_diagnoses": ["mastitis", "myiasis", "hematoma"],
    },
    "skin_infection_dermatitis": {
        "common_name": "Dermatitis (General Skin Inflammation)",
        "urgency": "LOW",
        "contagious": False,
        "zoonotic": False,
        "clinical_signs": [
            "Red, inflamed skin",
            "Itching and discomfort",
            "May progress to crusts, scabs, or weeping lesions",
            "Common in skin folds and contact areas",
        ],
        "immediate_actions": [
            "Identify and remove trigger (contact allergen, plant, chemical)",
            "Clean affected area with mild antiseptic soap",
            "Keep skin dry and ventilated",
        ],
        "management": [
            "Topical corticosteroid cream for inflammation",
            "Antibiotic ointment if secondary bacterial infection",
            "Antihistamines for severe allergic cases",
            "Dietary review (deficiencies can cause dermatitis)",
        ],
        "prevention": [
            "Identify and avoid allergens",
            "Good ventilation in housing",
            "Balanced nutrition with adequate zinc and vitamins",
            "Regular grooming",
        ],
        "when_to_call_vet": "If not improving in 7 days, spreading rapidly, or animal showing systemic signs",
        "differential_diagnoses": ["mange", "ringworm", "dermatophilosis"],
    },
    "skin_infection_dermatophilosis": {
        "common_name": "Dermatophilosis (Rain Scald)",
        "urgency": "MODERATE",
        "contagious": True,
        "zoonotic": True,
        "clinical_signs": [
            "Crusty, scabby lesions on back, neck, legs",
            "Matted hair tufts ('paintbrush' appearance)",
            "Worse in wet/humid conditions",
            "Underlying skin pink and moist when crusts removed",
        ],
        "immediate_actions": [
            "Move animal to dry shelter immediately",
            "Gently remove loose crusts (do not force)",
            "Wash with antiseptic shampoo (chlorhexidine or iodine-based)",
            "Wear gloves (zoonotic)",
        ],
        "management": [
            "Topical antibiotic spray or ointment daily",
            "Systemic antibiotics (penicillin) for severe cases",
            "Keep skin dry until fully healed",
            "Vitamin A/zinc supplementation supports skin healing",
        ],
        "prevention": [
            "Provide adequate dry shelter especially during rainy season",
            "Avoid prolonged exposure to wet conditions",
            "Treat ticks and biting insects (entry points for bacteria)",
            "Quarantine affected animals",
        ],
        "when_to_call_vet": "If lesions widespread, animal lethargic, or no improvement in 7 days",
        "differential_diagnoses": ["mange", "ringworm", "contagious ecthyma"],
    },
    "viral_orf": {
        "common_name": "Orf (Contagious Ecthyma)",
        "urgency": "MODERATE",
        "contagious": True,
        "zoonotic": True,
        "clinical_signs": [
            "Proliferative scabby lesions on lips, muzzle, nostrils",
            "Sometimes on udder, feet, or genitals",
            "Reduced feed intake due to mouth pain",
            "Lesions evolve: papule → vesicle → pustule → scab",
        ],
        "immediate_actions": [
            "Isolate affected animal (HIGHLY contagious)",
            "Wear gloves and long sleeves (ZOONOTIC — causes painful hand lesions in humans)",
            "Apply soothing ointment (glycerin or honey) to lesions",
            "Ensure soft, palatable feed (oats, mashed grain) and clean water",
        ],
        "management": [
            "Self-limiting in healthy adults (resolves in 3-4 weeks)",
            "Topical antiseptic to prevent secondary infection",
            "Antibiotics only if bacterial complication occurs",
            "Supportive care: soft food, electrolytes if dehydrated",
        ],
        "prevention": [
            "Live attenuated orf vaccine available in endemic areas",
            "Quarantine new arrivals 30 days",
            "Disinfect feeding troughs (virus survives in dry scabs for months)",
            "Strict hand hygiene after handling",
        ],
        "when_to_call_vet": "If animal stops eating, lesions become infected, or young kids affected",
        "differential_diagnoses": ["PPR", "FMD", "bluetongue"],
    },
    "viral_ppr": {
        "common_name": "Peste des Petits Ruminants (PPR / Goat Plague)",
        "urgency": "CRITICAL",
        "contagious": True,
        "zoonotic": False,
        "clinical_signs": [
            "High fever (40-41°C)",
            "Oral lesions, erosions on gums and tongue",
            "Mucopurulent nasal and ocular discharge",
            "Severe diarrhea, dehydration",
            "Pneumonia, labored breathing",
            "Mortality rate 50-90% in unvaccinated herds",
        ],
        "immediate_actions": [
            "Isolate ALL affected animals immediately",
            "Notify veterinary authorities — PPR is a NOTIFIABLE disease (WOAH listed)",
            "DO NOT move animals off premises",
            "Restrict farm visitors",
        ],
        "management": [
            "No specific antiviral treatment available",
            "Supportive care: fluids, electrolytes",
            "Antibiotics to prevent secondary bacterial pneumonia",
            "Anti-diarrheal and anti-inflammatory drugs",
            "Survivors develop lifelong immunity",
        ],
        "prevention": [
            "VACCINATION is the cornerstone (live attenuated PPR vaccine, single dose)",
            "Annual booster in endemic areas",
            "Quarantine all new animals for 21+ days",
            "Strict biosecurity at farm entrance",
        ],
        "when_to_call_vet": "IMMEDIATELY — PPR is a regulatory disease and requires official intervention",
        "differential_diagnoses": ["orf", "FMD", "bluetongue", "pasteurellosis"],
    },
    "wounds_footrot": {
        "common_name": "Footrot",
        "urgency": "HIGH",
        "contagious": True,
        "zoonotic": False,
        "clinical_signs": [
            "Severe lameness (often non-weight bearing)",
            "Foul odor from hoof",
            "Swelling between claws (interdigital)",
            "Under-running of hoof horn",
            "Black, necrotic tissue with discharge",
        ],
        "immediate_actions": [
            "Isolate from healthy animals (highly contagious in wet conditions)",
            "Trim affected hoof to expose all infected tissue",
            "Foot bath with 10% zinc sulfate or 5% formalin",
            "Apply topical antibiotic spray (oxytetracycline)",
        ],
        "management": [
            "Daily foot baths until healed",
            "Systemic antibiotics for severe cases (long-acting oxytetracycline)",
            "Keep animal on dry, clean bedding",
            "Pain management with NSAIDs",
            "Recheck and re-trim every 3-5 days",
        ],
        "prevention": [
            "Routine hoof trimming every 2-3 months",
            "Avoid wet, muddy environments",
            "Regular foot baths in endemic areas",
            "Cull chronically affected animals",
        ],
        "when_to_call_vet": "If lameness severe (animal can't walk), multiple feet affected, or no improvement in 7 days",
        "differential_diagnoses": ["foot abscess", "laminitis", "FMD"],
    },
}


def get_recommendation(disease_class, confidence=None):
    """Get treatment recommendation for a predicted disease class."""
    info = DISEASE_DB.get(disease_class)
    if not info:
        return f"No information available for class: {disease_class}"
    
    out = []
    out.append("=" * 70)
    out.append(f"CapraCare Treatment Recommendation")
    out.append("=" * 70)
    out.append(f"\nDetected: {info['common_name']}")
    out.append(f"Urgency: {info['urgency']}")
    out.append(f"Contagious: {'YES' if info['contagious'] else 'No'}")
    out.append(f"Zoonotic (transmissible to humans): {'YES — TAKE PRECAUTIONS' if info['zoonotic'] else 'No'}")
    
    if confidence is not None:
        if confidence < 0.60:
            out.append(f"\n⚠️  LOW CONFIDENCE ({confidence:.1%}) — recommend veterinary examination before treatment")
        elif confidence < 0.80:
            out.append(f"\n⚠️  MODERATE CONFIDENCE ({confidence:.1%}) — verify diagnosis if symptoms unclear")
        else:
            out.append(f"\n✓ Confidence: {confidence:.1%}")
    
    out.append(f"\n--- Clinical Signs to Verify ---")
    for s in info['clinical_signs']:
        out.append(f"  • {s}")
    
    out.append(f"\n--- Immediate Actions ---")
    for s in info['immediate_actions']:
        out.append(f"  • {s}")
    
    out.append(f"\n--- Ongoing Management ---")
    for s in info['management']:
        out.append(f"  • {s}")
    
    out.append(f"\n--- Prevention ---")
    for s in info['prevention']:
        out.append(f"  • {s}")
    
    out.append(f"\n--- When to Call a Veterinarian ---")
    out.append(f"  {info['when_to_call_vet']}")
    
    out.append(f"\n--- Differential Diagnoses (consider these too) ---")
    out.append(f"  {', '.join(info['differential_diagnoses'])}")
    
    out.append("\n" + "=" * 70)
    out.append("Sources: FAO, WOAH (World Organisation for Animal Health), Merck Veterinary Manual")
    out.append("DISCLAIMER: This system supports but does not replace veterinary diagnosis.")
    out.append("=" * 70)
    
    return "\n".join(out)


if __name__ == "__main__":
    # Demo each class
    for disease in DISEASE_DB.keys():
        print(get_recommendation(disease, confidence=0.92))
        print("\n")