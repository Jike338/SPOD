"""Length-bucketed microbatches; preserve per-action-state and per-task loss weights."""
def batches(samples, microbatch, token_budget):
    ordered=sorted(samples,key=lambda s:len(s['prompt'])+len(s['tokens']))
    current=[]
    for s in ordered:
        length=len(s['prompt'])+len(s['tokens'])
        if current and (len(current)>=microbatch or (len(current)+1)*length>token_budget):
            yield current;current=[]
        current.append(s)
    if current:yield current

def inputs(samples,pad,torch):
    sizes=[len(s['prompt'])+len(s['tokens']) for s in samples]
    ids=torch.full((len(samples),max(sizes)),pad,dtype=torch.long,device='cuda')
    mask=torch.zeros_like(ids)
    for i,s in enumerate(samples):
        ids[i,:sizes[i]]=torch.tensor(s['prompt']+s['tokens'],device='cuda')
        mask[i,:sizes[i]]=1
    return dict(input_ids=ids,attention_mask=mask,use_cache=False)
