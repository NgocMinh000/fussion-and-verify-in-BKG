import requests
import time
import hashlib
import pickle
import re
from datetime import datetime, timedelta
from collections import defaultdict
import random

# UMLS API credentials
API_KEY = '740425a9-0034-4363-8e9e-109bdc612ef4'

# Progress tracking
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.start_time = time.time()
        
    def update(self, entity_name, status):
        self.current += 1
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current
        remaining = self.total - self.current
        eta_seconds = avg_time * remaining
        eta = timedelta(seconds=int(eta_seconds))
        
        percentage = (self.current / self.total) * 100
        
        # Clear line và print progress
        print(f"\r[{self.current}/{self.total}] {percentage:.1f}% | "
              f"ETA: {eta} | {entity_name[:40]:<40} | {status}", 
              end='', flush=True)

def sample_kg_smart(input_file='kg.txt', output_file='kg_sample.txt', 
                    target_entities=3000, seed_ratio=0.3):
    """
    Sample KG với entity constraint
    Strategy: 30% high-degree + 70% random
    """
    print("🔄 Reading kg.txt...")
    
    all_triples = []
    entity_degree = defaultdict(int)
    entity_to_triples = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                head = parts[0].strip()
                relation = parts[1].strip()
                tail = parts[2].strip()
                
                triple = (head, relation, tail, line)
                all_triples.append(triple)
                
                entity_degree[head] += 1
                entity_degree[tail] += 1
                entity_to_triples[head].append(triple)
                entity_to_triples[tail].append(triple)
    
    print(f"📊 Total in kg.txt: {len(all_triples):,} triples, {len(entity_degree):,} entities")
    
    # Hybrid selection
    print(f"🎯 Sampling strategy: {int(seed_ratio*100)}% high-degree + {int((1-seed_ratio)*100)}% random...")
    
    sorted_entities = sorted(entity_degree.items(), key=lambda x: x[1], reverse=True)
    all_entities = [e for e, d in sorted_entities]
    
    num_high_degree = int(target_entities * seed_ratio)
    num_random = target_entities - num_high_degree
    
    # Lấy high-degree entities
    high_degree_entities = set(all_entities[:num_high_degree])
    
    # Lấy random từ phần còn lại
    remaining_entities = [e for e in all_entities if e not in high_degree_entities]
    if len(remaining_entities) >= num_random:
        random_entities = set(random.sample(remaining_entities, num_random))
    else:
        random_entities = set(remaining_entities)
    
    selected_entities = high_degree_entities | random_entities
    
    print(f"✅ Selected: {len(high_degree_entities)} high-degree + {len(random_entities)} random = {len(selected_entities)} entities")
    
    # Lấy triples với cả head và tail trong selected_entities
    selected_triples = set()
    for entity in selected_entities:
        for triple in entity_to_triples[entity]:
            head, rel, tail, line = triple
            if head in selected_entities and tail in selected_entities:
                selected_triples.add(line)
    
    selected_triples_list = list(selected_triples)
    
    # Lưu file
    with open(output_file, 'w') as f:
        f.writelines(selected_triples_list)
    
    # Verify statistics
    final_entities = set()
    final_relations = set()
    for line in selected_triples_list:
        parts = line.strip().split(',')
        if len(parts) == 3:
            final_entities.add(parts[0].strip())
            final_entities.add(parts[2].strip())
            final_relations.add(parts[1].strip())
    
    print(f"✅ Created: {output_file}")
    print(f"📊 Sample stats: {len(final_entities):,} entities, {len(final_relations):,} relations, {len(selected_triples_list):,} triples\n")
    
    return output_file

def get_umls_cui_with_type(term, api_key, retry=3):
    """Query UMLS API với retry"""
    search_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    
    for attempt in range(retry):
        try:
            search_params = {
                'string': term,
                'apiKey': api_key,
                'returnIdType': 'code',
                'pageSize': 5
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('result', {}).get('results', [])
                
                if results:
                    best_result = find_best_match(term, results)
                    cui = best_result['ui']
                    
                    cui_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
                    cui_params = {'apiKey': api_key}
                    
                    time.sleep(0.2)
                    cui_response = requests.get(cui_url, params=cui_params, timeout=10)
                    
                    if cui_response.status_code == 200:
                        cui_data = cui_response.json()
                        semantic_types = cui_data.get('result', {}).get('semanticTypes', [])
                        
                        if semantic_types:
                            sem_type_full = semantic_types[0].get('name', 'Entity')
                            sem_type_abbr = get_semantic_type_abbreviation(sem_type_full)
                            return cui, sem_type_abbr, 'UMLS'
                        else:
                            return cui, 'enty', 'UMLS'
                
                # Thử với simplified term
                simplified_term = simplify_term(term)
                if simplified_term != term:
                    return get_umls_cui_with_type_simplified(simplified_term, api_key)
                    
            elif response.status_code == 401:
                return None, None, 'API_ERROR'
            
        except requests.exceptions.Timeout:
            if attempt < retry - 1:
                time.sleep(1)
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(1)
    
    return None, None, 'NOT_FOUND'

def get_umls_cui_with_type_simplified(term, api_key):
    """Tìm kiếm với term đơn giản hóa"""
    search_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    search_params = {
        'string': term,
        'apiKey': api_key,
        'returnIdType': 'code',
        'pageSize': 3
    }
    
    try:
        response = requests.get(search_url, params=search_params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get('result', {}).get('results', [])
            
            if results:
                cui = results[0]['ui']
                
                cui_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
                cui_params = {'apiKey': api_key}
                time.sleep(0.2)
                cui_response = requests.get(cui_url, params=cui_params, timeout=10)
                
                if cui_response.status_code == 200:
                    cui_data = cui_response.json()
                    semantic_types = cui_data.get('result', {}).get('semanticTypes', [])
                    
                    if semantic_types:
                        sem_type_full = semantic_types[0].get('name', 'Entity')
                        sem_type_abbr = get_semantic_type_abbreviation(sem_type_full)
                        return cui, sem_type_abbr, 'UMLS_SIMPLIFIED'
    except Exception as e:
        pass
    
    return None, None, 'NOT_FOUND'

def find_best_match(term, results):
    """Tìm kết quả match tốt nhất"""
    term_lower = term.lower()
    
    for result in results:
        if result.get('name', '').lower() == term_lower:
            return result
    
    for result in results:
        if term_lower in result.get('name', '').lower():
            return result
    
    return results[0]

def simplify_term(term):
    """Đơn giản hóa term"""
    simplified = re.sub(r'\s+', ' ', term)
    simplified = simplified.strip()
    
    replacements = {
        'iuds': 'IUD',
        'atpase': 'ATPase',
    }
    
    for old, new in replacements.items():
        if old in simplified.lower():
            simplified = simplified.replace(old, new)
    
    return simplified

def infer_semantic_type_from_context(term):
    """Suy ra semantic type từ context"""
    term_lower = term.lower()
    
    protein_keywords = ['protein', 'enzyme', 'receptor', 'transporter', 'channel', 
                        'atpase', 'kinase', 'phosphatase', 'factor', 'antibody',
                        'immunoglobulin', 'cytokine', 'chemokine']
    if any(kw in term_lower for kw in protein_keywords):
        return 'aapp'
    
    enzyme_keywords = ['enzyme', 'ase', 'oxidase', 'reductase', 'transferase',
                       'hydrolase', 'lyase', 'isomerase', 'ligase']
    if any(kw in term_lower for kw in enzyme_keywords):
        return 'enzy'
    
    element_keywords = ['copper', 'iron', 'zinc', 'calcium', 'sodium', 'potassium',
                        'metal', 'ion', 'element', 'magnesium', 'selenium']
    if any(kw in term_lower for kw in element_keywords):
        return 'elmt'
    
    organic_keywords = ['acid', 'compound', 'molecule', 'chemical', 'lipid', 'carbohydrate']
    if any(kw in term_lower for kw in organic_keywords):
        return 'orch'
    
    device_keywords = ['device', 'iud', 'implant', 'catheter', 'stent', 'prosthesis']
    if any(kw in term_lower for kw in device_keywords):
        return 'medd'
    
    anatomy_keywords = ['cell', 'tissue', 'organ', 'body part', 'gland', 'membrane',
                        'macrophage', 'lymphocyte', 'neuron', 'muscle']
    if any(kw in term_lower for kw in anatomy_keywords):
        return 'cell' if 'cell' in term_lower else 'bpoc'
    
    process_keywords = ['uptake', 'transport', 'absorption', 'secretion', 'metabolism',
                        'synthesis', 'degradation', 'signaling', 'regulation']
    if any(kw in term_lower for kw in process_keywords):
        return 'biof'
    
    disease_keywords = ['disease', 'disorder', 'syndrome', 'infection', 'cancer',
                        'diabetes', 'hypertension', 'deficiency']
    if any(kw in term_lower for kw in disease_keywords):
        return 'dsyn'
    
    return 'enty'

def get_semantic_type_abbreviation(semantic_type_name):
    """Map semantic type sang abbreviation"""
    semantic_type_map = {
        'Pharmacologic Substance': 'phsu',
        'Antibiotic': 'antb',
        'Organic Chemical': 'orch',
        'Inorganic Chemical': 'inch',
        'Element, Ion, or Isotope': 'elmt',
        'Hormone': 'horm',
        'Enzyme': 'enzy',
        'Vitamin': 'vita',
        'Body Part, Organ, or Organ Component': 'bpoc',
        'Tissue': 'tisu',
        'Cell': 'cell',
        'Gene or Genome': 'gngm',
        'Amino Acid, Peptide, or Protein': 'aapp',
        'Receptor': 'rcpt',
        'Immunologic Factor': 'imft',
        'Disease or Syndrome': 'dsyn',
        'Mental or Behavioral Dysfunction': 'mobd',
        'Neoplastic Process': 'neop',
        'Sign or Symptom': 'sosy',
        'Injury or Poisoning': 'inpo',
        'Therapeutic or Preventive Procedure': 'topp',
        'Diagnostic Procedure': 'diap',
        'Laboratory Procedure': 'lbpr',
        'Medical Device': 'medd',
        'Research Device': 'resd',
        'Manufactured Object': 'mnob',
        'Substance': 'sbst',
        'Activity': 'acty',
        'Molecular Function': 'moft',
        'Biologic Function': 'biof',
        'Conceptual Entity': 'cnce',
        'Idea or Concept': 'idcn',
        'Qualitative Concept': 'qlco',
        'Quantitative Concept': 'qnco',
        'Entity': 'enty'
    }
    
    if semantic_type_name in semantic_type_map:
        return semantic_type_map[semantic_type_name]
    
    for key, value in semantic_type_map.items():
        if key.lower() in semantic_type_name.lower():
            return value
    
    return 'enty'

def generate_fallback_cui(term, semantic_type):
    """Generate CUI cho entities không có trong UMLS"""
    hash_id = int(hashlib.md5(term.encode()).hexdigest()[:7], 16) % 10000000
    return f"C{hash_id:07d}_{semantic_type}"

def main():
    from sklearn.model_selection import train_test_split
    
    print("="*80)
    print("UMLS Entity Mapper - Smart Sampling + Real-time Writing")
    print("="*80 + "\n")
    
    # Step 1: Sample KG với entity constraint
    kg_file = sample_kg_smart('kg.txt', 'kg_sample.txt', target_entities=3000, seed_ratio=0.3)
    
    print("="*80 + "\n")
    
    # Step 2: Đọc entities và relations từ kg_sample.txt
    entities = set()
    relations = set()
    triples = []
    
    print("🔄 Loading sampled KG...")
    with open(kg_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                head = parts[0].strip()
                relation = parts[1].strip()
                tail = parts[2].strip()
                entities.add(head)
                entities.add(tail)
                relations.add(relation)
                triples.append((head, relation, tail))
    
    print(f"✅ Loaded: {len(entities)} entities, {len(relations)} relations, {len(triples)} triples\n")
    
    # Step 3: Khởi tạo mapping dictionaries
    entity_mapping = {}
    entity2index = {}
    index2entity = {}
    relation2index = {}
    index2relation = {}
    
    # Step 4: Map relations
    print("🔄 Mapping relations...")
    for idx, relation in enumerate(sorted(relations)):
        rel_normalized = relation.upper().replace(' ', '_')
        relation2index[rel_normalized] = idx
        index2relation[idx] = rel_normalized
    
    with open('relation2index.pkl', 'wb') as f:
        pickle.dump(relation2index, f)
    with open('index2relation.pkl', 'wb') as f:
        pickle.dump(index2relation, f)
    
    print(f"✅ Saved: relation2index.pkl, index2relation.pkl ({len(relation2index)} relations)\n")
    
    # Step 5: Map entities với UMLS
    print("🔄 Mapping entities to UMLS CUI codes...")
    print(f"⏱️  Estimated time: ~{len(entities) * 0.5 / 60:.1f} minutes\n")
    
    tracker = ProgressTracker(len(entities))
    stats = {'umls': 0, 'generated': 0}
    
    # Mở file để ghi incremental
    mapping_file = open('entity_mapping.txt', 'w')
    mapping_file.write("# Entity\tCUI_Code\tSource\n")
    
    for idx, entity in enumerate(sorted(entities)):
        cui, sem_type, source = get_umls_cui_with_type(entity, API_KEY)
        
        if cui and sem_type:
            cui_full = f"{cui}_{sem_type}"
            stats['umls'] += 1
            status = "✓ UMLS"
        else:
            inferred_type = infer_semantic_type_from_context(entity)
            cui_full = generate_fallback_cui(entity, inferred_type)
            stats['generated'] += 1
            status = "⚠ Gen"
        
        entity_mapping[entity] = cui_full
        entity2index[cui_full] = idx
        index2entity[idx] = cui_full
        
        # Ghi vào file ngay
        mapping_file.write(f"{entity}\t{cui_full}\t{source}\n")
        mapping_file.flush()
        
        # Update progress
        tracker.update(entity, status)
        time.sleep(0.3)
    
    mapping_file.close()
    print("\n")
    
    # Step 6: Lưu entity mappings
    print("💾 Saving entity mappings...")
    with open('entity2index.pkl', 'wb') as f:
        pickle.dump(entity2index, f)
    with open('index2entity.pkl', 'wb') as f:
        pickle.dump(index2entity, f)
    
    print(f"✅ Saved: entity2index.pkl, index2entity.pkl ({len(entity2index)} entities)")
    print(f"✅ Saved: entity_mapping.txt\n")
    
    # Step 7: Statistics
    print("="*80)
    print("📊 Mapping Statistics:")
    print(f"   ✓ Found in UMLS: {stats['umls']} ({stats['umls']/len(entities)*100:.1f}%)")
    print(f"   ⚠ Generated CUI: {stats['generated']} ({stats['generated']/len(entities)*100:.1f}%)")
    print("="*80 + "\n")
    
    # Step 8: Convert triples
    print("🔄 Converting triples to train/valid/test...")
    converted_triples = []
    
    for head, relation, tail in triples:
        head_cui = entity_mapping[head]
        tail_cui = entity_mapping[tail]
        rel_normalized = relation.upper().replace(' ', '_')
        converted_triples.append(f"{head_cui}\t{rel_normalized}\t{tail_cui}")
    
    # Split
    train, temp = train_test_split(converted_triples, test_size=0.2, random_state=42, shuffle=True)
    valid, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)
    
    # Save
    with open('train.csv', 'w') as f:
        f.write('\n'.join(train))
    with open('valid.csv', 'w') as f:
        f.write('\n'.join(valid))
    with open('test.csv', 'w') as f:
        f.write('\n'.join(test))
    
    print(f"✅ Saved: train.csv ({len(train)} triples)")
    print(f"✅ Saved: valid.csv ({len(valid)} triples)")
    print(f"✅ Saved: test.csv ({len(test)} triples)\n")
    
    # Step 9: Summary
    print("="*80)
    print("✨ ALL DONE!")
    print("="*80)
    print(f"\n📁 Generated files:")
    print(f"   • kg_sample.txt (sampled data)")
    print(f"   • entity2index.pkl")
    print(f"   • index2entity.pkl")
    print(f"   • relation2index.pkl")
    print(f"   • index2relation.pkl")
    print(f"   • entity_mapping.txt")
    print(f"   • train.csv")
    print(f"   • valid.csv")
    print(f"   • test.csv")
    
    print(f"\n📝 Sample entity mappings:")
    for entity, cui in list(entity_mapping.items())[:5]:
        print(f"   {entity[:50]:<50} -> {cui}")
    
    print(f"\n📝 Sample relation mappings:")
    for rel, idx in list(relation2index.items())[:5]:
        print(f"   {rel:<30} -> {idx}")

if __name__ == "__main__":
    main()