import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterModule, Router } from '@angular/router';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterModule],
  templateUrl: './register.component.html',
  styleUrl: './register.component.css'
})
export class RegisterComponent {
  fullName: string = '';
  companyName: string = '';
  companyId: string = '';
  taxId: string = '';
  vatId: string = '';
  address: string = '';
  email: string = '';
  phone: string = '';
  username: string = '';
  password: string = '';
  repeatPassword: string = '';
  showPassword: boolean = false;
  showRepeatPassword: boolean = false;
  showSuccessPopup: boolean = false;
  
  consentPersonalData: boolean = true;
  consentTerms: boolean = false;
  consentMarketing: boolean = false;

  constructor(private router: Router) {}

  togglePasswordVisibility() {
    this.showPassword = !this.showPassword;
  }

  toggleRepeatPasswordVisibility() {
    this.showRepeatPassword = !this.showRepeatPassword;
  }

  onRegister() {
    // Registration logic will be implemented here
    console.log('Registration attempt:', { 
      fullName: this.fullName,
      email: this.email 
    });
    
    // Show success popup
    this.showSuccessPopup = true;
    
    // Hide popup and redirect after 5 seconds
    setTimeout(() => {
      this.showSuccessPopup = false;
      this.router.navigate(['/login']);
    }, 5000);
  }
}

