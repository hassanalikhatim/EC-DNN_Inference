import numpy as np



def compute_accuracies(
    outputs, decrypted_outputs, ground_truth
):
    
    outputs = np.array( [output_[-1] for output_ in outputs] )[:, 0]
    decrypted_outputs = np.array( [output_[-1] for output_ in decrypted_outputs] )[:, 0]
    
    assert outputs.shape == ground_truth.shape
    assert decrypted_outputs.shape == ground_truth.shape
    
    org_loss = np.mean( (outputs-ground_truth)**2 )
    org_acc = np.mean( np.argmax(outputs, axis=-1) == np.argmax(ground_truth, axis=-1) )
    
    enc_loss = np.mean( (decrypted_outputs-ground_truth)**2 )
    enc_acc = np.mean( np.argmax(ground_truth, axis=-1) == np.argmax(decrypted_outputs, axis=-1) )
    
    consistency = np.mean( np.argmax(outputs, axis=-1) == np.argmax(decrypted_outputs, axis=-1) )
    
    return (org_loss, org_acc), (enc_loss, enc_acc), consistency


def compute_differences(
    outputs, decrypted_outputs
):
    
    all_differences = []
    for k in range(len(outputs[0])):
        differences = 0
        for i in range(len(outputs)):
            differences += np.mean((decrypted_outputs[i][k].transpose().reshape(-1)-outputs[i][k].reshape(-1))**2)
        all_differences.append(differences)
    
    return all_differences
