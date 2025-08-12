import numpy as np
from enum import Enum, auto
from PIL import Image
import io

import matplotlib.pyplot as plt

class Mode(Enum):
    """
    Enumeration for different plotting modes in RTTrainingPlotter.
    
    Attributes:
        LOSS: For plotting only loss metrics
        ACCURACY: For plotting only accuracy metrics
        BOTH: For plotting both loss and accuracy metrics
    """
    LOSS = auto()
    ACCURACY = auto()
    BOTH = auto()
    
    @classmethod
    def from_string(cls, mode_str):
        """
        Convert a string to corresponding Mode enum value.
        
        Parameters:
        -----------
        mode_str : str
            String representation of mode ("loss", "accuracy", or "both")
            
        Returns:
        --------
        Mode
            Corresponding Mode enum value
        
        Raises:
        -------
        ValueError
            If mode_str is not a valid mode
        """
        mode_str = mode_str.lower()
        if mode_str == "loss":
            return cls.LOSS
        elif mode_str == "accuracy":
            return cls.ACCURACY
        elif mode_str == "both":
            return cls.BOTH
        else:
            raise ValueError(f"Invalid mode: {mode_str}. Must be 'loss', 'accuracy', or 'both'")

def plot_training_curve(train_metrics, test_metrics=None, metric_name="Loss", 
                        title="Training Curve", save_path=None):
    """
    Visualize training and testing curves.
    
    Parameters:
    -----------
    train_metrics : list or numpy.ndarray
        Values of the metric during training
    test_metrics : list or numpy.ndarray, optional
        Values of the metric during testing/validation
    metric_name : str, default="Loss"
        Name of the metric being plotted
    title : str, default="Training Curve"
        Title of the plot
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_metrics) + 1)
    
    # Plot training metrics
    plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    
    # Plot test metrics if provided
    if test_metrics is not None:
        plt.plot(epochs, test_metrics, 'r-', label=f'Validation {metric_name}')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add markers at min/max points for better visibility
    if test_metrics is not None:
        best_epoch = np.argmin(test_metrics) + 1 if "loss" in metric_name.lower() else np.argmax(test_metrics) + 1
        plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, 
                    label=f'Best epoch ({best_epoch})')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

class RTTrainingPlotter:
    """
    A class for real-time visualization of training metrics.
    
    Allows updating the plot with each epoch's metrics and supports
    loss, accuracy, and combined visualization modes.
    
    @Example:
    ```
    plotter = RTTrainingPlotter(mode=Mode.LOSS)
    for epoch in range(num_epochs):
        train_loss = train_epoch()
        val_loss = validate()
        plotter.update(epoch, train_loss, val_loss)
    plotter.show()
    """
    
    def __init__(self, mode:Mode = Mode.LOSS, figsize=(10, 6), save_path=None, save_as_gif=False):
        """
        Initialize the real-time training plotter.
        
        Parameters:
        -----------
        mode : Mode, default=Mode.LOSS
            Mode to use for plotting (Mode.LOSS, Mode.ACCURACY, or Mode.BOTH)
        figsize : tuple, default=(10, 6)
            Size of the figure in inches
        save_path : str, optional
            Path to save the final plot (PNG) or animated GIF
        save_as_gif : bool, default=False
            Whether to save all updates as an animated GIF
        """
        # Convert string to Mode enum if string is provided
        if isinstance(mode, str):
            self.mode = Mode.from_string(mode)
        else:
            self.mode = mode
        
        # Set up data containers
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.epochs = []
        
        # Save parameters
        self.save_path = save_path
        self.save_as_gif = save_as_gif
        self.frames = [] if save_as_gif else None
        
        # Initialize figure
        plt.ion()  # Turn on interactive mode
        
        if self.mode == Mode.BOTH:
            self.fig = plt.figure(figsize=(15, 6))
            # Create loss subplot
            self.ax_loss = self.fig.add_subplot(121)
            self.train_loss_line, = self.ax_loss.plot([], [], 'b-', label='Training Loss')
            self.val_loss_line, = self.ax_loss.plot([], [], 'r-', label='Validation Loss')
            self.ax_loss.set_title("Real-time Training and Validation Loss")
            self.ax_loss.set_xlabel("Epochs")
            self.ax_loss.set_ylabel("Loss")
            self.ax_loss.grid(True, linestyle='--', alpha=0.7)
            self.ax_loss.legend()
            self.best_loss_line = None
            
            # Create accuracy subplot
            self.ax_acc = self.fig.add_subplot(122)
            self.train_acc_line, = self.ax_acc.plot([], [], 'b-', label='Training Accuracy')
            self.val_acc_line, = self.ax_acc.plot([], [], 'r-', label='Validation Accuracy')
            self.ax_acc.set_title("Real-time Training and Validation Accuracy")
            self.ax_acc.set_xlabel("Epochs")
            self.ax_acc.set_ylabel("Accuracy")
            self.ax_acc.grid(True, linestyle='--', alpha=0.7)
            self.ax_acc.legend()
            self.best_acc_line = None
        else:
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(111)
            
            # Configure based on mode
            if self.mode == Mode.LOSS:
                self.ylabel = "Loss"
                self.find_best = np.argmin
            elif self.mode == Mode.ACCURACY:
                self.ylabel = "Accuracy"
                self.find_best = np.argmax
            
            # Create empty line objects
            self.train_line, = self.ax.plot([], [], 'b-', label=f'Training {self.ylabel}')
            self.val_line, = self.ax.plot([], [], 'r-', label=f'Validation {self.ylabel}')
            self.best_line = None
            
            # Set up the plot
            self.ax.set_title(f"Real-time Training and Validation {self.ylabel}")
            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel(self.ylabel)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.legend()
        
        plt.tight_layout()
    
    def update(self, epoch, train_metric, val_metric=None, train_acc=None, val_acc=None):
        """
        Update the plot with new metrics.
        
        Parameters:
        -----------
        epoch : int
            Current epoch number
        train_metric : float
            Training loss value for the current epoch
        val_metric : float, optional
            Validation loss value for the current epoch
        train_acc : float, optional
            Training accuracy value (required when mode is Mode.BOTH)
        val_acc : float, optional
            Validation accuracy value (required when mode is Mode.BOTH)
        """
        # Add epoch to epochs
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        
        if self.mode == Mode.BOTH:
            # Ensure we have accuracy data
            if train_acc is None or val_acc is None:
                raise ValueError("train_acc and val_acc must be provided when mode is Mode.BOTH")
            
            # Update loss data
            self.train_loss.append(train_metric)
            self.train_loss_line.set_data(self.epochs, self.train_loss)
            
            # Update accuracy data
            self.train_acc.append(train_acc)
            self.train_acc_line.set_data(self.epochs, self.train_acc)
            
            # Update validation metrics if provided
            if val_metric is not None:
                self.val_loss.append(val_metric)
                self.val_loss_line.set_data(self.epochs, self.val_loss)
                
                self.val_acc.append(val_acc)
                self.val_acc_line.set_data(self.epochs, self.val_acc)
            
            # Adjust the view for loss plot
            self.ax_loss.set_xlim(0, max(epoch+1, 10))
            ymin_loss = 0
            ymax_loss = max(self.train_loss + (self.val_loss if val_metric is not None else [])) * 1.1
            self.ax_loss.set_ylim(ymin_loss, ymax_loss)
            
            # Adjust the view for accuracy plot
            self.ax_acc.set_xlim(0, max(epoch+1, 10))
            ymin_acc = min(0, min(self.train_acc + (self.val_acc if val_acc is not None else [])) * 0.9)
            ymax_acc = max(1.0, max(self.train_acc + (self.val_acc if val_acc is not None else [])) * 1.1)
            self.ax_acc.set_ylim(ymin_acc, ymax_acc)
            
            # Update best lines if validation metrics exist
            if val_metric is not None and len(self.val_loss) > 0:
                # Update best loss epoch
                if self.best_loss_line:
                    self.best_loss_line.remove()
                
                best_loss_idx = np.argmin(self.val_loss)
                best_loss_epoch = self.epochs[best_loss_idx]
                self.best_loss_line = self.ax_loss.axvline(x=best_loss_epoch, color='g', linestyle='--', alpha=0.5,
                            label=f'Best epoch ({best_loss_epoch})')
                self.ax_loss.legend()
                
                # Update best accuracy epoch
                if self.best_acc_line:
                    self.best_acc_line.remove()
                
                best_acc_idx = np.argmax(self.val_acc)
                best_acc_epoch = self.epochs[best_acc_idx]
                self.best_acc_line = self.ax_acc.axvline(x=best_acc_epoch, color='g', linestyle='--', alpha=0.5,
                            label=f'Best epoch ({best_acc_epoch})')
                self.ax_acc.legend()
        else:
            # Add new data for single metric mode
            if self.mode == Mode.LOSS:
                self.train_loss.append(train_metric)
                self.train_line.set_data(self.epochs, self.train_loss)
                metrics = self.train_loss
                val_metrics = self.val_loss
            else:  # Mode.ACCURACY
                self.train_acc.append(train_metric)
                self.train_line.set_data(self.epochs, self.train_acc)
                metrics = self.train_acc
                val_metrics = self.val_acc
            
            # Update validation line if provided
            if val_metric is not None:
                if self.mode == Mode.LOSS:
                    self.val_loss.append(val_metric)
                    self.val_line.set_data(self.epochs, self.val_loss)
                else:  # Mode.ACCURACY
                    self.val_acc.append(val_metric)
                    self.val_line.set_data(self.epochs, self.val_acc)
            
            # Adjust the view as more data comes in
            self.ax.set_xlim(0, max(epoch+1, 10))
            
            # Calculate y-limits based on available data
            all_metrics = metrics.copy()
            if val_metric is not None:
                all_metrics.extend(val_metrics)
            
            if self.mode == Mode.LOSS:
                ymin = 0
                ymax = max(all_metrics) * 1.1
            else:  # Mode.ACCURACY
                ymin = min(0, min(all_metrics) * 0.9) if all_metrics else 0
                ymax = max(1.0, max(all_metrics) * 1.1) if all_metrics else 1.0
                
            self.ax.set_ylim(ymin, ymax)
            
            # Update best epoch line if validation metrics exist
            if val_metric is not None and len(val_metrics) > 0:
                # Remove old vertical line if it exists
                if self.best_line:
                    self.best_line.remove()
                    self.best_line = None
                
                # Find best epoch so far
                best_epoch_idx = self.find_best(val_metrics)
                best_epoch = self.epochs[best_epoch_idx]
                
                # Add vertical line for best epoch
                self.best_line = self.ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5,
                              label=f'Best epoch ({best_epoch})')
                
                # Update legend
                self.ax.legend()
        
        # Update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Capture frame for GIF if requested
        if self.save_as_gif:
            # Need to import here to avoid dependency for users not using this feature

            # Convert figure canvas to image
            buf = io.BytesIO()
            self.fig.canvas.print_png(buf)
            buf.seek(0)
            img = Image.open(buf)
            # Create a copy of the image before adding to frames
            img_copy = img.copy()
            self.frames.append(img_copy)
            buf.close()
            img.close()
        
    def show(self):
        """
        Display the final plot in non-interactive mode and save if requested.
        """
        plt.ioff()
        
        # Save final image as PNG if requested
        if self.save_path and not self.save_as_gif:
            self.fig.savefig(self.save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {self.save_path}")
        
        # Save animated GIF if requested
        if self.save_path and self.save_as_gif and self.frames:
            try:
                # Ensure the save_path has .gif extension
                gif_path = self.save_path
                if not gif_path.lower().endswith('.gif'):
                    gif_path += '.gif'
                
                # Save frames as animated GIF
                self.frames[0].save(
                    gif_path,
                    format='GIF',
                    append_images=self.frames[1:],
                    save_all=True,
                    duration=200,  # 200ms per frame
                    loop=0  # Loop forever
                )
                print(f"Animated GIF saved to {gif_path}")
            except Exception as e:
                print(f"Error saving GIF: {e}")
        
        plt.show()
        
    def close(self):
        """
        Close the plot.
        """
        plt.close(self.fig)
    
    def get_best_epoch(self, metric_type = None) -> tuple:
        """
        Get the epoch number with the best validation metric.
        
        Parameters:
        -----------
        metric_type : Mode or str, optional
            Specify which metric to use (Mode.LOSS or Mode.ACCURACY) when mode is Mode.BOTH
            
        Returns:
        --------
        tuple
            (epoch number, metric value) with the best validation metric
        """
        if self.mode == Mode.BOTH and metric_type is None:
            raise ValueError("Must specify metric_type (Mode.LOSS or Mode.ACCURACY) when mode is Mode.BOTH")
        
        # Convert string to Mode enum if string is provided
        if isinstance(metric_type, str):
            metric_type = Mode.from_string(metric_type)
        
        if self.mode == Mode.BOTH:
            if metric_type == Mode.LOSS:
                if len(self.val_loss) == 0:
                    raise ValueError("Validation loss metrics are not available")
                best_epoch_idx = np.argmin(self.val_loss)
                return self.epochs[best_epoch_idx], self.val_loss[best_epoch_idx]
            elif metric_type == Mode.ACCURACY:
                if len(self.val_acc) == 0:
                    raise ValueError("Validation accuracy metrics are not available")
                best_epoch_idx = np.argmax(self.val_acc)
                return self.epochs[best_epoch_idx], self.val_acc[best_epoch_idx]
        else:
            val_metrics = self.val_loss if self.mode == Mode.LOSS else self.val_acc
            if len(val_metrics) == 0:
                raise ValueError(f"Validation {self.mode.name.lower()} metrics are not available")
            
            best_epoch_idx = self.find_best(val_metrics)
            return self.epochs[best_epoch_idx], val_metrics[best_epoch_idx]

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    epochs = 100
    train_loss = np.random.rand(epochs) * np.exp(-np.arange(epochs) * 0.05) + 0.1
    val_loss = train_loss + 0.05 * np.random.randn(epochs) + 0.1
    
    # Plot and show
    # plot_training_curve(train_loss, val_loss, metric_name="Loss", 
    #                     title="Model Training and Validation Loss")
    # plt.show()
    
    # # Example with accuracy
    # train_acc = 1 - train_loss/2
    # val_acc = 1 - val_loss/2
    # plot_training_curve(train_acc, val_acc, metric_name="Accuracy", 
    #                   title="Model Training and Validation Accuracy")
    # plt.show()
    
    # Real-time training curve visualization
    plotter = RTTrainingPlotter(metric_type="accuracy")
    for epoch in range(epochs):
        plotter.update(epoch, train_loss[epoch], val_loss[epoch])
        plt.pause(0.1)
    plotter.show()