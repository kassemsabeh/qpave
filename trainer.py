import time

from tqdm import tqdm
import torch

# Trainer class to manage training and log metrics
class QATrainer():
  def __init__(self, train_loader, validation_loader, model, optimizer, loss_function, config, device):
    self.config = config
    self.epochs = self.config['epochs']
    self.train_loader = train_loader
    self.validation_loader = validation_loader
    self.model = model
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.log_every = self.config['log_every']
    self.alpha = self.config['alpha']
    self.device = device
    self.n_steps = 0
    self.tolerance = 2
    self.min_loss = 50

  # Function to save torch model checkpoint
  def __save_model(self, path):
    torch.save(self.model.state_dict(), path)
  

  # Function to calculate the accuracy on the qa task (strict cirteria)
  def __calculate_qa_accuracy(self, predicted_start, predicted_end, target_start, target_end):
    a = (predicted_start.argmax(1).cpu() == target_start.cpu())
    b = (predicted_end.argmax(1).cpu() == target_end.cpu())
    return torch.logical_and(a, b).sum().item()
  
  # Function to calculate the accuracy on the category classifcation task
  def __calculate_category_accuracy(self, predicted_category, target_category):
    val, idx = torch.max(predicted_category.data, dim=1)
    return (idx==target_category).sum().item()

  # Function to train one epoch of the model
  def __train_epoch(self, epoch):
    self.model.train()
    total_qa_correct, total_category_correct, total_count, total_loss = 0, 0, 0, 0
    epoch_qa_correct, epoch_category_correct, n_examples = 0, 0, 0
    start_time = time.time()

    for idx, data in enumerate(tqdm(self.train_loader)):
      self.n_steps += 1
      # Read the input data
      ids = data['ids'].to(self.device, dtype=torch.long)
      mask = data['mask'].to(self.device, dtype=torch.long)
      original_attribute = data['original_attributes'].to(self.device, dtype=torch.long)
      target_start = data['start_pos'].to(self.device, dtype=torch.long)
      target_end = data['end_pos'].to(self.device, dtype=torch.long)
      target_category = data['targets'].to(self.device, dtype=torch.long)
      # Forward pass
      self.optimizer.zero_grad()
      predicted_start, predicted_end, predicted_category = self.model(ids, mask, original_attribute)

      # Calculate the loss and accuracy
      start_loss = self.loss_function(predicted_start, target_start)
      end_loss = self.loss_function(predicted_end, target_end)
      total_count += ids.size(0)
      n_examples += total_count
      self.batch_count += 1
      total_qa_correct += self.__calculate_qa_accuracy(predicted_start, predicted_end, target_start, target_end)
      epoch_qa_correct += total_qa_correct
      if self.config['use_categories']:
        category_loss = self.loss_function(predicted_category, target_category)
        loss = self.alpha * (start_loss + end_loss) + (1 - (2 * self.alpha)) * category_loss
        total_category_correct += self.__calculate_category_accuracy(predicted_category, target_category)
        epoch_category_correct += total_category_correct
      else:
        loss = 0.5 * (start_loss + end_loss)
      total_loss += loss.item()

      # Run the backward pass
      loss.backward()

      # Update the parameters of the model
      self.optimizer.step()
     # Log the results
      if (self.n_steps % self.log_every == 0 ) and self.n_steps > 0:
        qa_accuracy = total_qa_correct / total_count
        category_accuracy = total_category_correct / total_count
        elapsed = time.time() - start_time
        print(f"| epoch {epoch:3d} | {idx + 1}/{len(self.train_loader)} batches | loss {loss.item():2f} | qa accuracy {qa_accuracy:.2f} | category accuracy {category_accuracy:.2f}")
        total_qa_correct = 0
        total_category_correct = 0
        total_count = 0
        start_time = time.time()
      
      # Return the total loss and accuracies
    return total_loss, (epoch_qa_correct / n_examples), (epoch_category_correct / n_examples)
  
  # Function to evaluate the model
  def evaluate(self):
    self.model.eval()
    total_loss, total_qa_correct, total_category_correct, total_count = 0, 0, 0, 0
    steps = 0
    with torch.no_grad():
      for idx, data in enumerate(self.validation_loader):
        steps += 1
        # Transform to torch tensors
        ids = data['ids'].to(self.device, dtype=torch.long)
        mask = data['mask'].to(self.device, dtype=torch.long)
        original_attribute = data['original_attributes'].to(self.device, dtype=torch.long)
        target_start = data['start_pos'].to(self.device, dtype=torch.long)
        target_end = data['end_pos'].to(self.device, dtype=torch.long)
        target_category = data['targets'].to(self.device, dtype=torch.long)
        total_count += target_category.size(0)
        # Calculate the output
        predicted_start, predicted_end, predicted_category = self.model(ids, mask, original_attribute)
        # Calculate the loss
        start_loss = self.loss_function(predicted_start, target_start)
        end_loss = self.loss_function(predicted_end, target_end)
        total_qa_correct += self.__calculate_qa_accuracy(predicted_start, predicted_end, target_start, target_end)
        if self.config['use_categories']:
          category_loss = self.loss_function(predicted_category, target_category)
          loss = self.alpha * (start_loss + end_loss) + (1 - (2 * self.alpha)) * category_loss
          # loss = start_loss + end_loss + category_loss

          total_category_correct += self.__calculate_category_accuracy(predicted_category, target_category)
        else:
          loss = 0.5 * (start_loss + end_loss)
        total_loss += loss.item()
    return (total_loss / steps), (total_qa_correct / total_count), (total_category_correct / total_count)

  # Function to train the model
  def train(self):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    # wandb.watch(self.model, self.loss_function, log="all", log_freq=10)
    # Start the training loop
    for epoch in range(1, self.epochs + 1):
      self.batch_count = 0
      epoch_start_time = time.time()
      loss, epoch_qa_accuracy, epoch_category_accuracy = self.__train_epoch(epoch)
      epoch_loss = loss / self.batch_count
      val_loss, val_qa_accuracy, val_category_accuracy = self.evaluate()
      elapsed_time = time.time() - epoch_start_time
      print('-' * 200)
      print(f'| end of epoch {epoch:3d} | time: {elapsed_time:5.2f}s | training loss: {epoch_loss:.2f} |  validation loss: {val_loss:.2f} | training qa accuracy: {epoch_qa_accuracy:.2f} | validation qa accuracy: {val_qa_accuracy:.2f} | training category accuracy: {epoch_category_accuracy:.2f} | validation category accuracy: {val_category_accuracy:.2f}')
      print('-' * 200)
      if self.config['save_model']:
          if val_loss < self.min_loss:
            print("Saving trained model in local repository..")
            self.__save_model(self.config['model_save_path'])
            self.min_loss = val_loss
          else:
            self.tolerance -= 1
      if self.tolerance == 0:
        print(f"Stopping training after {epoch} epochs")
        break
      
